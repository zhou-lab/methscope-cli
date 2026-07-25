// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_DIGEST_H
#define METHSCOPE_DIGEST_H

#include <stddef.h>

/* Lowercase hex SHA-256 of a buffer / of a file. `out` takes 64 chars + NUL.
 * ms_sha256_file returns 0 on success, -1 if the file cannot be read. */
void ms_sha256_buf(const void *data, size_t len, char out[65]);
int ms_sha256_file(const char *path, char out[65]);

#endif
