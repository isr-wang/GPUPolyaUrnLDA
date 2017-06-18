OS = $(shell uname -s)

NVCC			= 	nvcc
FLAGS 		+= 	-std=c++11
ifdef DEBUG
FLAGS		+=	-ggdb -O0
else
FLAGS		+=	-O2
endif

LIBS 		= 	
CFLAGS		+=	-I. -I./include $(FLAGS) -c -Wno-deprecated-gpu-targets
LDFLAGS		+=	$(FLAGS)

SOURCES		= 	GPUPolyaUrnLDA.cxx LDA.cxx
OBJECTS		= 	$(SOURCES:.cxx=.o)
EXECUTABLE	=	GPUPolyaUrnLDA

.PHONY: all clean

all: $(EXECUTABLE)

%.o: %.cxx
	$(NVCC) $< -o $@ $(CFLAGS)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LDFLAGS) -Wno-deprecated-gpu-targets

clean:
	rm -rv $(EXECUTABLE) $(OBJECTS) $(DEPENDENCIES)
