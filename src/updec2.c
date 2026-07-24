// SPDX-License-Identifier: AGPL-3.0-or-later
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include "updec2.h"

static int uerr(char *dst, size_t cap, const char *msg) {
  if (cap) snprintf(dst, cap, "%s", msg);
  return 0;
}

static int span(uint64_t off, uint64_t count, uint64_t size, uint64_t limit) {
  return off <= limit && (!size || count <= (limit - off) / size);
}

int ms_updec2_open(ms_updec2_t *m, const char *path,
                   uint64_t offset, uint64_t length,
                   char *error, size_t error_cap) {
  memset(m, 0, sizeof(*m)); m->fd = -1;
  int fd = open(path, O_RDONLY);
  if (fd < 0) return uerr(error, error_cap, "cannot open UPDEC2");
  struct stat st;
  if (fstat(fd, &st) || st.st_size < 0) {
    close(fd); return uerr(error, error_cap, "cannot stat UPDEC2");
  }
  uint64_t file_bytes = (uint64_t)st.st_size;
  if (!length) length = file_bytes - offset;
  if (offset > file_bytes || length > file_bytes - offset ||
      length < sizeof(ms_updec2_header_t)) {
    close(fd); return uerr(error, error_cap, "UPDEC2 section is out of bounds");
  }
  if (file_bytes > SIZE_MAX) {
    close(fd); return uerr(error, error_cap, "UPDEC2 mapping exceeds address space");
  }
  void *map = mmap(NULL, (size_t)file_bytes, PROT_READ, MAP_SHARED, fd, 0);
  if (map == MAP_FAILED) {
    close(fd); return uerr(error, error_cap, "cannot mmap UPDEC2");
  }
  const unsigned char *base = (const unsigned char *)map + offset;
  const ms_updec2_header_t *h = (const ms_updec2_header_t *)base;
  if (memcmp(h->magic, MS_UPDEC2_MAGIC, 8) ||
      (h->version != 2 && h->version != 3)) {
    munmap(map, (size_t)file_bytes); close(fd);
    return uerr(error, error_cap, "bad UPDEC2 magic or version");
  }
  if (sizeof(*h) != 128 || sizeof(ms_updec2_unit_t) != 40 ||
      sizeof(ms_updec2_membership_t) != 24) {
    munmap(map, (size_t)file_bytes); close(fd);
    return uerr(error, error_cap, "UPDEC2 ABI layout mismatch");
  }
  uint64_t prep_count = h->version == 2 ? h->patterns : h->input_dim;
  uint32_t trunk_dim =
    h->version == 3 && (h->flags & MS_UPDEC2_FLAG_TRUNK)
      ? (uint32_t)h->reserved0 : 0;
  uint32_t expected_input =
    h->version == 3 && (h->flags & MS_UPDEC2_FLAG_BETA_ONLY)
      ? h->patterns : 2 * h->patterns;
  if (!h->patterns || h->patterns > UINT32_MAX / 2 ||
      h->input_dim != expected_input || !h->n_units ||
      !h->n_cpg || h->n_cpg > UINT32_MAX || h->file_bytes != length ||
      !span(h->mean_offset, prep_count, sizeof(float), length) ||
      !span(h->scale_offset, prep_count, sizeof(float), length) ||
      !span(h->unit_offset, h->n_units, sizeof(ms_updec2_unit_t), length) ||
      !span(h->cpg_offset, h->n_cpg, sizeof(uint32_t), length) ||
      !span(h->membership_offset, h->n_memberships,
            sizeof(ms_updec2_membership_t), length) ||
      h->param_offset > length ||
      (h->version == 2 && (h->flags & ~MS_UPDEC2_FLAG_GENOMIC)) ||
      (h->version == 3 && (h->flags & ~(MS_UPDEC2_FLAG_GENOMIC |
          MS_UPDEC2_FLAG_COUNT | MS_UPDEC2_FLAG_TRUNK |
          MS_UPDEC2_FLAG_BETA_ONLY))) ||
      ((h->flags & MS_UPDEC2_FLAG_COUNT) &&
       (h->flags & MS_UPDEC2_FLAG_BETA_ONLY)) ||
      ((h->flags & MS_UPDEC2_FLAG_TRUNK) && !trunk_dim) ||
      (!(h->flags & MS_UPDEC2_FLAG_TRUNK) && h->reserved0)) {
    munmap(map, (size_t)file_bytes); close(fd);
    return uerr(error, error_cap, "invalid UPDEC2 header offsets");
  }
  const float *mean = (const float *)(base + h->mean_offset);
  const float *scale = (const float *)(base + h->scale_offset);
  for (uint64_t p = 0; p < prep_count; ++p) {
    if (!isfinite(mean[p]) || !isfinite(scale[p]) || !(scale[p] > 0.0f)) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "invalid UPDEC2 preprocessing");
    }
  }
  uint64_t trunk_floats = 0, unit_input = h->input_dim;
  if (trunk_dim) {
    uint64_t I = h->input_dim, H = trunk_dim;
    if (H > UINT32_MAX || H > (UINT64_MAX - 2 * H) / (I + H)) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "UPDEC2 trunk dimensions overflow");
    }
    trunk_floats = H * I + H + H * H + H;
    if (!span(h->param_offset, trunk_floats, sizeof(float), length)) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "UPDEC2 trunk is out of bounds");
    }
    unit_input = H;
  }
  uint64_t first_unit_param = h->param_offset + trunk_floats * sizeof(float);
  const ms_updec2_unit_t *u = (const ms_updec2_unit_t *)(base + h->unit_offset);
  uint64_t expected = 0;
  for (uint32_t k = 0; k < h->n_units; ++k) {
    if (u[k].output_offset != expected || !u[k].cpg_count ||
        u[k].output_offset > h->n_cpg ||
        u[k].cpg_count > h->n_cpg - u[k].output_offset ||
        u[k].param_offset < first_unit_param ||
        u[k].param_offset > length ||
        u[k].param_bytes > length - u[k].param_offset ||
        (u[k].mode == MS_UPDEC2_FACTOR && !u[k].bottleneck_dim) ||
        (u[k].mode == MS_UPDEC2_DIRECT && u[k].bottleneck_dim) ||
        u[k].mode > MS_UPDEC2_FACTOR) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "invalid UPDEC2 unit directory");
    }
    uint64_t floats;
    if (u[k].mode == MS_UPDEC2_FACTOR) {
      uint64_t r = u[k].bottleneck_dim, I = unit_input, O = u[k].cpg_count;
      floats = r * I + r + O * r + O;
    } else {
      floats = (uint64_t)u[k].cpg_count * unit_input + u[k].cpg_count;
    }
    if (floats > UINT64_MAX / sizeof(float) ||
        u[k].param_bytes != floats * sizeof(float)) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "UPDEC2 unit parameter size mismatch");
    }
    expected += u[k].cpg_count;
  }
  if (expected != h->n_cpg || h->activation > MS_UPDEC2_LEAKY_RELU) {
    munmap(map, (size_t)file_bytes); close(fd);
    return uerr(error, error_cap, "UPDEC2 coverage or activation is invalid");
  }
  const uint32_t *cpg = (const uint32_t *)(base + h->cpg_offset);
  size_t seen_bytes = (size_t)((h->n_cpg + 7) >> 3);
  unsigned char *seen = calloc(seen_bytes ? seen_bytes : 1, 1);
  if (!seen) {
    munmap(map, (size_t)file_bytes); close(fd);
    return uerr(error, error_cap, "out of memory validating UPDEC2 CpGs");
  }
  for (uint64_t q = 0; q < h->n_cpg; ++q) {
    uint32_t pos = cpg[q];
    if (pos >= h->n_cpg || (seen[pos >> 3] & (1u << (pos & 7)))) {
      free(seen); munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "UPDEC2 CpG map is not a permutation");
    }
    seen[pos >> 3] |= (unsigned char)(1u << (pos & 7));
  }
  free(seen);
  const ms_updec2_membership_t *members =
    (const ms_updec2_membership_t *)(base + h->membership_offset);
  for (uint32_t q = 0; q < h->n_memberships; ++q) {
    if (members[q].unit >= h->n_units || !members[q].count ||
        members[q].output_offset > h->n_cpg ||
        members[q].count > h->n_cpg - members[q].output_offset) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "invalid UPDEC2 membership descriptor");
    }
    const ms_updec2_unit_t *mu = &u[members[q].unit];
    if (members[q].output_offset < mu->output_offset ||
        members[q].output_offset + members[q].count >
          mu->output_offset + mu->cpg_count) {
      munmap(map, (size_t)file_bytes); close(fd);
      return uerr(error, error_cap, "UPDEC2 membership crosses a unit boundary");
    }
  }
  m->fd = fd; m->mapping = map; m->mapping_bytes = (size_t)file_bytes;
  m->model_offset = offset; m->model_bytes = length; m->base = base; m->header = h;
  m->mean = mean; m->scale = scale; m->units = u;
  m->cpg = cpg;
  m->memberships = members;
  m->trunk_dim = trunk_dim;
  if (trunk_dim) {
    const float *p = (const float *)(base + h->param_offset);
    m->trunk_w1 = p;
    m->trunk_b1 = p + (uint64_t)trunk_dim * h->input_dim;
    m->trunk_w2 = m->trunk_b1 + trunk_dim;
    m->trunk_b2 = m->trunk_w2 + (uint64_t)trunk_dim * trunk_dim;
  }
  return 1;
}

void ms_updec2_close(ms_updec2_t *m) {
  if (m->mapping) munmap(m->mapping, m->mapping_bytes);
  if (m->fd >= 0) close(m->fd);
  memset(m, 0, sizeof(*m)); m->fd = -1;
}

void ms_updec2_prepare_input(const ms_updec2_t *m, const double *beta,
                             const int *count, float *x) {
  uint32_t P = m->header->patterns;
  if (m->header->version == 2) {
    for (uint32_t p = 0; p < P; ++p) {
      if (isnan(beta[p])) {
        x[p] = 0.0f;
        x[P + p] = 1.0f;
      } else {
        x[p] = (float)((beta[p] - m->mean[p]) / m->scale[p]);
        x[P + p] = 0.0f;
      }
    }
    return;
  }
  if (m->header->flags & MS_UPDEC2_FLAG_BETA_ONLY) {
    for (uint32_t p = 0; p < P; ++p)
      x[p] = isnan(beta[p]) || (count && count[p] == 0)
        ? 0.0f : (float)((beta[p] - m->mean[p]) / m->scale[p]);
    return;
  }
  for (uint32_t p = 0; p < P; ++p) {
    uint32_t j = 2 * p;
    int n = count ? count[p] : (isnan(beta[p]) ? 0 : 1);
    x[j] = isnan(beta[p]) || n == 0
      ? 0.0f : (float)((beta[p] - m->mean[j]) / m->scale[j]);
    double aux = (m->header->flags & MS_UPDEC2_FLAG_COUNT)
      ? log1p((double)(n > 0 ? n : 0)) : (n == 0 ? 1.0 : 0.0);
    x[j + 1] = (float)((aux - m->mean[j + 1]) / m->scale[j + 1]);
  }
}

static float sigmoidf_stable(float x) {
  if (x >= 0.0f) {
    float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  float z = expf(x);
  return z / (1.0f + z);
}

int ms_updec2_forward(const ms_updec2_t *m, const float *x, float *prob,
                      float *work, size_t workcap) {
  const ms_updec2_header_t *h = m->header;
  uint32_t I = h->input_dim;
  const float *unit_x = x;
  float *z = work;
  size_t zcap = workcap;
  if (m->trunk_dim) {
    uint32_t H = m->trunk_dim;
    if (workcap < (size_t)2 * H) return 0;
    float *h1 = work, *h2 = work + H;
    for (uint32_t r = 0; r < H; ++r) {
      double acc = m->trunk_b1[r];
      const float *w = m->trunk_w1 + (uint64_t)r * I;
      for (uint32_t j = 0; j < I; ++j) acc += (double)w[j] * x[j];
      if (acc < 0.0) acc *= 0.01;
      h1[r] = (float)acc;
    }
    for (uint32_t r = 0; r < H; ++r) {
      double acc = m->trunk_b2[r];
      const float *w = m->trunk_w2 + (uint64_t)r * H;
      for (uint32_t j = 0; j < H; ++j) acc += (double)w[j] * h1[j];
      if (acc < 0.0) acc *= 0.01;
      h2[r] = h1[r] + (float)acc;
    }
    unit_x = h2; I = H; z = work + 2 * H; zcap = workcap - 2 * H;
  }
  for (uint32_t ui = 0; ui < h->n_units; ++ui) {
    const ms_updec2_unit_t *u = &m->units[ui];
    const float *par = (const float *)(m->base + u->param_offset);
    const uint32_t O = u->cpg_count;
    if (u->mode == MS_UPDEC2_FACTOR) {
      const uint32_t R = u->bottleneck_dim;
      if (zcap < R) return 0;
      const float *A = par;
      const float *a = A + (uint64_t)R * I;
      const float *E = a + R;
      const float *b = E + (uint64_t)O * R;
      for (uint32_t r = 0; r < R; ++r) {
        double acc = a[r];
        const float *w = A + (uint64_t)r * I;
        for (uint32_t j = 0; j < I; ++j) acc += (double)w[j] * unit_x[j];
        if (h->activation == MS_UPDEC2_LEAKY_RELU && acc < 0.0) acc *= 0.01;
        z[r] = (float)acc;
      }
      for (uint32_t o = 0; o < O; ++o) {
        double acc = b[o];
        const float *w = E + (uint64_t)o * R;
        for (uint32_t r = 0; r < R; ++r) acc += (double)w[r] * z[r];
        prob[m->cpg[u->output_offset + o]] = sigmoidf_stable((float)acc);
      }
    } else {
      const float *W = par;
      const float *b = W + (uint64_t)O * I;
      for (uint32_t o = 0; o < O; ++o) {
        double acc = b[o];
        const float *w = W + (uint64_t)o * I;
        for (uint32_t j = 0; j < I; ++j) acc += (double)w[j] * unit_x[j];
        prob[m->cpg[u->output_offset + o]] = sigmoidf_stable((float)acc);
      }
    }
  }
  return 1;
}
