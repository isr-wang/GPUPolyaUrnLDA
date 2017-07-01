OS = $(shell uname -s)

NVCC		= 	nvcc
FLAGS 		+= 	-std=c++11
ifdef DEBUG
FLAGS		+=	-ggdb -O0
else
FLAGS		+=	-O2
endif

LIBS 		= 	
CFLAGS		+=	-I. -I./include $(FLAGS) -c -Wno-deprecated-gpu-targets
LDFLAGS		+=	$(FLAGS)

rwildcard=$(wildcard $(addsuffix $2, $1)) $(foreach d,$(wildcard $(addsuffix *, $1)),$(call rwildcard,$d/,$2)) 

SOURCES		:= 	$(call rwildcard,./,*.cu)
OBJECTS		= 	$(SOURCES:.cu=.o)
EXECUTABLE	=	GPUPolyaUrnLDA

.PHONY: all clean

all: $(EXECUTABLE)

%.o: %.cu
	$(NVCC) $< -o $@ $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LDFLAGS) -Wno-deprecated-gpu-targets

clean:
	rm -rv $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES)

# We want to exclude 3rd-party headers from being formatted
format-all:
	python ../tools/format.py --format --exclude-pattern '.*/include/.*'

# Only format chunks that have changed
format:
	git diff -U0 HEAD^ | clang-format-diff -style=file -i -p1
