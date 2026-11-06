// Prints a serialized STRELA executable as JSON, e.g. one written by
//   iree-compile --iree-hal-dump-executable-binaries-to=<dir> ...
//
// Usage: estela-dump-executable [--header] <file.str1>
//
// The file is an IREE header (iree_flatbuffer_file_header_t) followed by an
// ExecutableDef FlatBuffer, see runtime/src/estela/schemas. Without --header
// only the FlatBuffer is printed; with it the output is
//   {"header": {...}, "executable": {...}}

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// NOTE: include order matters.
#include "iree/base/internal/flatcc/debugging.h"
#include "strela_executable_def_json_printer.h"
#include "strela_executable_def_reader.h"
#include "strela_executable_def_verifier.h"

static char *
read_file(const char *path, size_t *out_size) {
  FILE *file = fopen(path, "rb");
  char *data = NULL;
  long size = -1;

  if (!file) return NULL;
  if (fseek(file, 0, SEEK_END) == 0) size = ftell(file);
  if (size >= 0 && fseek(file, 0, SEEK_SET) == 0) {
    data = malloc(size ? (size_t)size : 1);
    if (data && fread(data, 1, (size_t)size, file) != (size_t)size) {
      free(data);
      data = NULL;
    }
  }
  fclose(file);
  *out_size = (size_t)size;
  return data;
}

int main(int argc, char **argv) {
  bool header = false;
  const char *path = NULL;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--header") == 0) {
      header = true;
    } else if (!path && argv[i][0] != '-') {
      path = argv[i];
    } else {
      path = NULL;
      break;
    }
  }
  if (!path) {
    fprintf(stderr, "usage: %s [--header] <file.str1>\n", argv[0]);
    return 2;
  }

  size_t size = 0;
  char *data = read_file(path, &size);
  if (!data) {
    fprintf(stderr, "%s: cannot read the file\n", path);
    return 1;
  }

  int status = 1;

  iree_flatbuffer_file_header_t file_header;
  const char *content = data + sizeof file_header;
  if (size < sizeof file_header) {
    fprintf(stderr, "%s: shorter than the %zu byte IREE header\n", path,
            sizeof file_header);
    goto done;
  }
  memcpy(&file_header, data, sizeof file_header);
  if (memcmp(&file_header.magic, iree_hal_strela_ExecutableDef_file_identifier,
             sizeof file_header.magic) != 0) {
    fprintf(stderr, "%s: not a STRELA executable (magic is not \"%s\")\n",
            path, iree_hal_strela_ExecutableDef_file_identifier);
    goto done;
  }
  if (file_header.content_size > size - sizeof file_header) {
    fprintf(stderr, "%s: the header says %llu content bytes, the file has %zu\n",
            path, (unsigned long long)file_header.content_size,
            size - sizeof file_header);
    goto done;
  }

  int error = iree_hal_strela_ExecutableDef_verify_as_root(
      content, (size_t)file_header.content_size);
  if (error != flatcc_verify_ok) {
    fprintf(stderr, "%s: invalid ExecutableDef: %s\n", path,
            flatcc_verify_error_string(error));
    goto done;
  }

  if (header) {
    printf("{\n\"header\": {\n"
           "  \"magic\": \"%.4s\",\n"
           "  \"version\": %u,\n"
           "  \"content_size\": %llu,\n"
           "  \"reserved\": [",
           (const char *)&file_header.magic, file_header.version,
           (unsigned long long)file_header.content_size);
    size_t reserved_count =
        sizeof file_header.reserved / sizeof file_header.reserved[0];
    for (size_t i = 0; i < reserved_count; ++i) {
      printf("%s%llu", i ? ", " : "",
             (unsigned long long)file_header.reserved[i]);
    }
    printf("]\n},\n\"executable\": ");
  }

  flatcc_json_printer_t printer;
  flatcc_json_printer_init(&printer, stdout);
  flatcc_json_printer_set_flags(&printer, flatcc_json_printer_f_pretty);
  // A column with word_count 0 (unused) is worth seeing.
  flatcc_json_printer_set_skip_default(&printer, false);
  flatcc_json_printer_set_force_default(&printer, true);
  strela_executable_def_print_json(&printer, content,
                                   (size_t)file_header.content_size);
  flatcc_json_printer_flush(&printer);
  error = flatcc_json_printer_get_error(&printer);
  flatcc_json_printer_clear(&printer);
  if (error) {
    fprintf(stderr, "%s: JSON printing failed (%d)\n", path, error);
    goto done;
  }

  if (header) printf("}\n");
  status = 0;

done:
  free(data);
  return status;
}
