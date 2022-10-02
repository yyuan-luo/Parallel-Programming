//
// Created by Dennis-Florian Herr on 13/06/2022.
//

#include <string>
#include <deque>
#include <future>
#include <functional>
#include <iostream>

#include "Utility.h"

#define MEASURE_TIME true
#define THREADS 16

int numProblems = 10000;
Sha1Hash* solutionHashes;

struct Problem {
    Sha1Hash sha1_hash;
    int problemNum;
};


class ProblemQueue {
public:
    void push(Problem problem){
        std::lock_guard<std::mutex> lock(mutex);
        problem_queue.push_back(problem);

        cv.notify_one();
    }

    Problem pop(){
        std::unique_lock<std::mutex> lock(mutex);
        while (problem_queue.empty())
            cv.wait(lock);
        Problem p = problem_queue.front();
        problem_queue.pop_front();
        return p;
    }

    bool empty(){
        return problem_queue.empty();
    }

private:
    std::deque<Problem> problem_queue;
    std::mutex mutex;
    std::condition_variable cv;

};

ProblemQueue problemQueue;

// generate numProblems sha1 hashes with leadingZerosProblem leading zero bits
// This method is intentionally compute intense so you can already start working on solving
// problems while more problems are generated
void generateProblem(int seed, int numProblems, int leadingZerosProblem){
    srand(seed);

    for(int i = 0; i < numProblems; i++){
        std::string base = std::to_string(rand()) + std::to_string(rand());
        Sha1Hash hash = Utility::sha1(base);
        do{
            // we keep hashing ourself until we find the desired amount of leading zeros
            hash = Utility::sha1(hash);
        }while(Utility::count_leading_zero_bits(hash) < leadingZerosProblem);
        problemQueue.push(Problem{hash, i});
    }
}

// This method repeatedly hashes itself until the required amount of leading zero bits is found
Sha1Hash findSolutionHash(Sha1Hash hash, int leadingZerosSolution){
    do{
        // we keep hashing ourself until we find the desired amount of leading zeros
        hash = Utility::sha1(hash);
    }while(Utility::count_leading_zero_bits(hash) < leadingZerosSolution);

    return hash;
}

void worker_thread(int leadingZerosSolution) {
    while(true) {
        Problem p = problemQueue.pop();
        if (p.problemNum < 0)
            break;
        solutionHashes[p.problemNum] = findSolutionHash(p.sha1_hash, leadingZerosSolution);
    }
}

int main(int argc, char *argv[]) {
    int leadingZerosProblem = 8;
    int leadingZerosSolution = 11;
    int numProblems = 10000;

    //Not interesting for parallelization
    Utility::parse_input(numProblems, leadingZerosProblem, leadingZerosSolution, argc, argv);

    solutionHashes = static_cast<Sha1Hash*>(malloc(numProblems * sizeof(Sha1Hash)));

    unsigned int seed = Utility::readInput();

    std::thread generatorThread(generateProblem, seed, numProblems, leadingZerosProblem);

#if MEASURE_TIME
    struct timespec generation_start, generation_end;
    clock_gettime(CLOCK_MONOTONIC, &generation_start);
#endif

//    generateProblem(seed, numProblems, leadingZerosProblem);
    std::thread threads[THREADS];

    for (int i = 0; i < THREADS; ++i) {
        threads[i] = std::thread(worker_thread, leadingZerosSolution);
    }

    generatorThread.join();

#if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &generation_end);
    double generation_time = (((double) generation_end.tv_sec + 1.0e-9 * generation_end.tv_nsec) - ((double) generation_start.tv_sec + 1.0e-9 * generation_start.tv_nsec));
    fprintf(stderr, "Generate Problem time:  %.7gs\n", generation_time);

    struct timespec solve_start, solve_end;
    clock_gettime(CLOCK_MONOTONIC, &solve_start);
#endif

    for (int i = 0; i < THREADS; ++i) {
        problemQueue.push(Problem{Sha1Hash(), -1});
    }

    for (int i = 0; i < THREADS; ++i) {
        threads[i].join();
    }

#if MEASURE_TIME
    clock_gettime(CLOCK_MONOTONIC, &solve_end);
    double solve_time = (((double) solve_end.tv_sec + 1.0e-9 * solve_end.tv_nsec) - ((double) solve_start.tv_sec + 1.0e-9 * solve_start.tv_nsec));
    fprintf(stderr, "Solve Problem time:     %.7gs\n", solve_time);
#endif


    Sha1Hash solution;
    // guarantee initial solution hash data is zero
    memset(solution.data, 0, SHA1_BYTES);
    // this doesn't need parallelization. it's neglectibly fast
    for(int i = 0; i < numProblems; i++){
        solution = Utility::sha1(solution, solutionHashes[i]);
    }

    Utility::printHash(solution);
//    for (int i = 0; i < numProblems; ++i) {
//        std::cout << i << " ";
//        Utility::printHash(solutionHashes[i]);
//        std::cout << std::endl;
//    }
    printf("DONE\n");

    return 0;
}
