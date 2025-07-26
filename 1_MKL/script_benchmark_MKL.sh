#!/bin/bash

# Get current directory name (used as the implementation name)
IMPL_NAME=$(basename "$(pwd)")

# Color codes
BLUE='\033[1;34m'
GREEN='\033[1;32m'
GRAY='\033[1;90m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color


echo -e "${YELLOW}⚠️  NumPy/PyTorch warnings are suppressed. Output from scripts will still be shown.${NC}"
echo ""

# Welcome message
echo -e "${BLUE}🚀 Starting Benchmark Suite for implementation: ${IMPL_NAME}${NC}"
echo -e "${GRAY}This will run benchmarking, profiling, and flamegraph generation for CUDA matmul kernel...${NC}"
echo ""

# Step 1: Benchmarking
echo -e "${GREEN}🔧 Step 1: Running Benchmark.py${NC}"
python Benchmark.py 2>/dev/null || { echo -e "${RED}✖ Benchmark.py failed${NC}"; exit 1; }
echo -e "${GRAY}✔ Benchmark completed.${NC}"
echo ""

# Step 2: Performance profiling summary
echo -e "${GREEN}📊 Step 2: Running ShowPerformance.py${NC}"
python ShowPerformance.py 2>/dev/null || { echo -e "${RED}✖ ShowPerformance.py failed${NC}"; exit 1; }
echo -e "${GRAY}✔ Profiling summary saved. You can open it in TensorBoard or check the .txt file.${NC}"
echo ""

# Step 3: Flamegraph trace
echo -e "${GREEN}🔥 Step 3: Running GenerateFlameGraph.py${NC}"
python GenerateFlameGraph.py 2>/dev/null || { echo -e "${RED}✖ GenerateFlameGraph.py failed${NC}"; exit 1; }
echo -e "${GRAY}✔ Flamegraph trace exported to JSON (for Chrome tracing).${NC}"
echo ""

# Final message
echo -e "${BLUE}✅ All steps completed for: ${IMPL_NAME}${NC}"
echo -e "${GRAY}You can now view performance results, open the trace in Chrome, or analyze the logs.${NC}"
