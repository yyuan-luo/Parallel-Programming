CXX = c++
CXX_FLAGS = --std=c++17 -Wall -Wextra -fopenmp -march=native -O3 -g

all: env_file sequential_implementation student_submission reference_implementation

sequential_implementation: Utility.h sequential_implementation.cpp
	$(CXX) $(CXX_FLAGS) -o sequential_implementation sequential_implementation.cpp 

student_submission: Utility.h student_submission.cpp 
	$(CXX) $(CXX_FLAGS) -o student_submission student_submission.cpp

reference_implementation: Utility.h reference_implementation.cpp 
	$(CXX) $(CXX_FLAGS) -o reference_implementation reference_implementation.cpp

env_file: student_submission.cpp
	sed -n -e '!submission_env \K.*' student_submission.cpp > student_submission.env; [ $$? -lt 2 ]

clean:
	rm -f sequential_implementation student_submission reference_implementation student_submission.env *.o
