# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2 -Iinclude -MMD -MP
AR = ar
ARFLAGS = rcs

# Directories
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib
BIN_DIR = bin
TEST_DIR = tests
INCLUDE_DIR = include

# Library name
LIBRARY = $(LIB_DIR)/libcmlib.a

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Test files
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_EXECUTABLES = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BIN_DIR)/%)
TEST_NAMES = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=%)

# Default target
all: $(LIBRARY) $(TEST_EXECUTABLES)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create static library
$(LIBRARY): $(OBJECTS)
	@mkdir -p $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^

# Compile and link test executables to bin directory
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(LIBRARY)
	@mkdir -p $(BIN_DIR) $(OBJ_DIR)/tests
	$(CXX) $(CXXFLAGS) -MF $(OBJ_DIR)/tests/$*.d $< -L$(LIB_DIR) -lcmlib -o $@

# Run the unit tests (no plotting, non-zero exit code on failure)
test: $(BIN_DIR)/unitTests
	./$(BIN_DIR)/unitTests

# run-<name> builds and runs tests/<name>.cpp, e.g. make run-transformerLM
run-%: $(BIN_DIR)/% | data/tmp
	./$(BIN_DIR)/$*
	rm -f data/tmp/*

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(BIN_DIR)
	rm -f data/tmp/*

# Create data directory if it doesn't exist
data/tmp:
	mkdir -p data/tmp

# Print variables for debugging
debug:
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "TEST_SOURCES: $(TEST_SOURCES)"
	@echo "TEST_EXECUTABLES: $(TEST_EXECUTABLES)"

.PHONY: all clean debug test
.SECONDARY: $(TEST_EXECUTABLES)

-include $(OBJECTS:.o=.d) $(TEST_NAMES:%=$(OBJ_DIR)/tests/%.d)
