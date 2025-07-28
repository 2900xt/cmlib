# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -Iinclude
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

# Default target
all: $(LIBRARY) $(TEST_EXECUTABLES)

# Create directories
$(OBJ_DIR) $(LIB_DIR) $(BIN_DIR):
	mkdir -p $@

$(OBJ_DIR)/math $(OBJ_DIR)/alg $(OBJ_DIR)/plot: | $(OBJ_DIR)
	mkdir -p $@

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)/math $(OBJ_DIR)/alg $(OBJ_DIR)/plot
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create static library
$(LIBRARY): $(OBJECTS) | $(LIB_DIR)
	$(AR) $(ARFLAGS) $@ $^

# Compile and link test executables to bin directory
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(LIBRARY) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -L$(LIB_DIR) -lcmlib -o $@

# Test targets - run the executables
run-plotTest: $(BIN_DIR)/plotTest | data/tmp
	./$(BIN_DIR)/plotTest
	rm -f data/tmp/*

run-linearReg: $(BIN_DIR)/linearReg | data/tmp
	./$(BIN_DIR)/linearReg
	rm -f data/tmp/*

run-logisticReg: $(BIN_DIR)/logisticReg | data/tmp
	./$(BIN_DIR)/logisticReg
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

.PHONY: all clean debug run-plotTest run-linearReg