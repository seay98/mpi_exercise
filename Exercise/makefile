# Examples/Makefile
# ============================================================
# Top-level Makefile that includes:
#  - utils/build.mk      (libraries: array, multiply)
#  - build.mk (exercises)
# ============================================================

# -------- project root (Examples directory) --------
PROJECT_ROOT := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))

# -------- build output dirs --------
BUILD_DIR   ?= $(PROJECT_ROOT)/build
OBJ_DIR     ?= $(BUILD_DIR)/obj
BIN_DIR     ?= $(BUILD_DIR)/bin
LIB_DIR     ?= $(BUILD_DIR)/lib
INCLUDE_DIR ?= $(PROJECT_ROOT)/include

# -------- toolchain --------
CC     ?= gcc
MPICC  ?= mpicc
AR     ?= ar

# NOTE: your libs.mk uses "$(AR) $(ARFLAGS) ..."
# so provide sane defaults (rcs recommended for static libs)
ARFLAGS ?= rcs

# -------- flags --------
CFLAGS  ?= -Og -Wall -g
LDFLAGS ?=

# -------- BLAS pkgconf package name --------
BLAS_LIB ?= openblas

# -------- header install helper used by libs.mk --------
# libs.mk expects: $(INSTALL_HEADER) <dest-dir> <src>
# We'll implement it as a "cp -f" with mkdir -p.
INSTALL_HEADER := install -m 0644 -Dt

# ============================================================
# Include library rules (array/multiply)
# ============================================================
# This file is the one you pasted ("LIBS := array multiply ...")
include $(PROJECT_ROOT)/utils/build.mk

# ============================================================
# Include exercises rules
# ============================================================
include $(PROJECT_ROOT)/build.mk

# ============================================================
# Directory creation
# ============================================================
$(BUILD_DIR) $(OBJ_DIR) $(BIN_DIR) $(LIB_DIR) $(INCLUDE_DIR):
	mkdir -p $@

# (libs.mk uses $(UTIL_OBJ_DIR) which is $(OBJ_DIR)/util)
$(OBJ_DIR)/util:
	mkdir -p $@

# Ensure exercises links succeed by building libs first
exercises: libraries

# Default target
.PHONY: all
all: libraries exercises

# ============================================================
# Convenience targets
# ============================================================
.PHONY: clean distclean
# clean:
# 	rm -rf $(OBJ_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: print-vars
print-vars:
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "BUILD_DIR=$(BUILD_DIR)"
	@echo "OBJ_DIR=$(OBJ_DIR)"
	@echo "BIN_DIR=$(BIN_DIR)"
	@echo "LIB_DIR=$(LIB_DIR)"
	@echo "INCLUDE_DIR=$(INCLUDE_DIR)"
	@echo "CC=$(CC)"
	@echo "MPICC=$(MPICC)"
	@echo "AR=$(AR)"
	@echo "ARFLAGS=$(ARFLAGS)"
	@echo "BLAS_LIB=$(BLAS_LIB)"
	@echo "EXERCISES_OBJ_DIR=$(EXERCISES_OBJ_DIR)"
	@echo "EXERCISES_SRC_DIR=$(EXERCISES_SRC_DIR)"

.PHONY: test
test:
	@echo "Running exercise_2_1"
	@$(BIN_DIR)/exercise_2_1 || echo ": Failed to run exercise_2_1, stat $$?"
