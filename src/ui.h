// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UI_H
#define METHSCOPE_UI_H

#include <stddef.h>

/* True only when stdin *and* stderr are terminals. Every prompt is gated on
 * this, so a pipeline or container build can never stop at a question. */
int ms_ui_interactive(void);

const char *ms_ui_dim(void);
const char *ms_ui_bold(void);
const char *ms_ui_green(void);
const char *ms_ui_cyan(void);
const char *ms_ui_reset(void);

/* Checkbox list drawn on stderr; returns a malloc'd flag per item, or NULL if
 * the user cancelled or there is no terminal. Falls back to a numbered list
 * read from one line when raw mode is unavailable. */
int *ms_ui_multiselect(const char *title, const char *const *items,
                       const char *const *notes, size_t n, int preselect);

/* Off a terminal this returns `default_yes` rather than asking. */
int ms_ui_confirm(const char *question, int default_yes);

#endif
