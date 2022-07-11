//
// Created by Vincent Bode on 08/07/2020.
//

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "life.h"
#include "VideoOutput.h"
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <random>
#include <unistd.h>

#define CODE_VERSION 1
//#define DEBUG 1

std::minstd_rand random_engine;
uint_fast32_t cached_value;
uint_fast32_t bit_mask = 0;

void seed_generator(unsigned long long seed) {
    random_engine = std::minstd_rand(seed);
}

inline bool generate_bit() {
    if (!bit_mask) {
        cached_value = random_engine();
        bit_mask = 1;
    }
    bool value = cached_value & bit_mask;
    bit_mask = bit_mask << 1;
    return value;
}

int compare_buff(bool *send, bool *receive) {
    for (int i = 0; i < 751; ++i) {
        if (send[i] != receive[i]) {
            return 0;
        }
    }
    return 1;
}

/*
  Apply the game of life rules on a Torus --> grid contains shadow rows and columns
  to simplify application of rules i.e. grid actually ranges from grid [ 1.. height - 2 ][ 1 .. width - 2]
*/
void evolve(ProblemData &problemData, int rank, int block_width) {
    auto &grid = *problemData.readGrid;
    auto &writeGrid = *problemData.writeGrid;

    // For each cell
    for (int i = std::max(1, rank * block_width);
         i < std::min(GRID_SIZE - 1, (rank + 1) * block_width); i++) {
        for (int j = 1; j < GRID_SIZE - 1; ++j) {
            // Calculate the number of neighbors
            int sum = grid[i - 1][j - 1] + grid[i - 1][j] + grid[i - 1][j + 1] +
                      grid[i][j - 1] + grid[i][j + 1] +
                      grid[i + 1][j - 1] + grid[i + 1][j] + grid[i + 1][j + 1];

            if (!grid[i][j]) {
                // If a cell is dead, it can start living by reproduction or stay dead
                if (sum == 3) {
                    // reproduction
                    writeGrid[i][j] = true;
                } else {
                    writeGrid[i][j] = false;
                }
            } else {
                // If a cell is alive, it can stay alive or die through under/overpopulation
                if (sum == 2 || sum == 3) {
                    // stays alive
                    writeGrid[i][j] = true;
                } else {
                    // dies due to under or overpopulation
                    writeGrid[i][j] = false;
                }
            }
        }
    }
}

/*
  Copies data from the inner part of the grid to
  shadow (padding) rows and columns to transform the grid into a torus.
*/
void copy_edges_mpi(bool(&grid)[GRID_SIZE][GRID_SIZE], int rank, int block_width, int size) {

    MPI_Status status;

    // horizontal
    {
        for (int i = rank * block_width; i < (rank + 1) * block_width; ++i) {
            grid[i][0] = grid[i][GRID_SIZE - 2];
            grid[i][GRID_SIZE - 1] = grid[i][1];
        }
    }

    // vertical
    {
        // sent to bottom, receive from top
        int des = rank + 1;
        int source = rank - 1;
        if (des == size)
            des = 0;
        if (source == -1)
            source = size - 1;

        MPI_Sendrecv(&grid[std::min(GRID_SIZE - 1, (rank + 1) * block_width) - 1][0], GRID_SIZE, MPI_CXX_BOOL, des, 0,
                     &grid[std::max(1, rank * block_width) - 1][0], GRID_SIZE, MPI_CXX_BOOL, source, 0,
                     MPI_COMM_WORLD, &status);

        // send to top, receive from bottom
        des = rank - 1;
        source = rank + 1;
        if (des == -1)
            des = size - 1;
        if (source == size)
            source = 0;
        MPI_Sendrecv(&grid[std::max(1, rank * block_width)][0], GRID_SIZE, MPI_CXX_BOOL, des, 1,
                     &grid[std::min(GRID_SIZE - 1, (rank + 1) * block_width)][0], GRID_SIZE, MPI_CXX_BOOL, source, 1,
                     MPI_COMM_WORLD, &status);
    }

    // corners
    {
        if (rank == 0) {
            MPI_Send(&grid[1][1], 1, MPI_CXX_BOOL, size - 1, 2, MPI_COMM_WORLD);
            MPI_Recv(&grid[0][0], 1, MPI_CXX_BOOL, size - 1, 3, MPI_COMM_WORLD, nullptr);

            MPI_Send(&grid[1][GRID_SIZE - 2], 1, MPI_CXX_BOOL, size - 1, 4, MPI_COMM_WORLD);
            MPI_Recv(&grid[0][GRID_SIZE - 1], 1, MPI_CXX_BOOL, size - 1, 5, MPI_COMM_WORLD, nullptr);
        } else if (rank == size - 1) {
            MPI_Recv(&grid[GRID_SIZE - 1][GRID_SIZE - 1], 1, MPI_CXX_BOOL, 0, 2, MPI_COMM_WORLD, nullptr);
            MPI_Send(&grid[GRID_SIZE - 2][GRID_SIZE - 2], 1, MPI_CXX_BOOL, 0, 3, MPI_COMM_WORLD);

            MPI_Recv(&grid[GRID_SIZE - 1][0], 1, MPI_CXX_BOOL, 0, 4, MPI_COMM_WORLD, nullptr);
            MPI_Send(&grid[GRID_SIZE - 2][1], 1, MPI_CXX_BOOL, 0, 5, MPI_COMM_WORLD);
        }
    }
}

int count_alive(ProblemData &data, int rank, int block_width) {
    auto &grid = *data.readGrid;
    int counter = 0;
    for (int i = std::max(1, rank * block_width);
         i < std::min(GRID_SIZE - 1, (rank + 1) * block_width); i++)
        for (int j = 1; j < GRID_SIZE - 1; ++j) {
            if (grid[i][j])
                counter++;
        }
    return counter;
}

void print_problem(ProblemData &problem, int rank, int block_width) {
    printf("rank: %d:\n", rank);
    auto &grid = *problem.readGrid;
//    for (int i = std::max(1, (rank / sqrt_size) * block_width);
//         i < std::min(GRID_SIZE - 1, (rank / sqrt_size + 1) * block_width); i++) {
//        for (int j = std::max(1, (rank % sqrt_size) * block_width);
//             j < std::min(GRID_SIZE - 1, (rank % sqrt_size + 1) * block_width); j++) {
//            if (grid[i][j])
//                printf("1 ");
//            else
//                printf("0 ");
//        }
//        printf("\n");
//    }
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            if (grid[i][j])
                printf("1 ");
            else
                printf("0 ");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    bool activateVideoOutput = false;
    if (argc > 1) {
        if (argc == 2 && strcmp(argv[1], "-g") == 0) {
            activateVideoOutput = true;
        } else {
            fprintf(stderr, "Usage:\n  %s [-g]\n    -g: Activate graphical output.\n", argv[0]);
            exit(153);
        }
    }

    auto *problemData = new ProblemData;
    auto &grid = *problemData->readGrid;

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int block_width = GRID_SIZE / size;

//    copy_edges(*problemData->readGrid);

    // As with Jack Sparrow's exercise, this needs FFMPEG (new and improved: this now works with more video players).
    // As an alternative, you can write individual png files to take a look at the data.

    unsigned int seed = 0;
    if (rank == 0) {
        std::cout << "READY" << std::endl;
        std::cin >> seed;

        std::cout << "Using seed " << seed << std::endl;
        if (seed == 0) {
            std::cout << "Warning: default value 0 used as seed." << std::endl;
        }
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    seed_generator(seed);

    for (int i = 0; i < GRID_SIZE * GRID_SIZE; i += 1) {
        *(grid[0] + i) = generate_bit();
    }

//    printf("rank: %d, x: %d-%d, y: %d-%d\n", rank,
//           std::max(1, (rank / sqrt_size) * block_width),
//           std::min(GRID_SIZE - 1, (rank / sqrt_size + 1) * block_width),
//
//           std::max(1, (rank % sqrt_size) * block_width),
//           std::min(GRID_SIZE - 1, (rank % sqrt_size + 1) * block_width));
    //TODO@Students: This is the main simulation. Parallelize it using MPI.
    for (int iteration = 0; iteration <= NUM_SIMULATION_STEPS; ++iteration) {
//        if (iteration == 0 && rank == 3) {
//            print_problem(*problemData, rank, block_width);
//            printf("\n");
//        }
        copy_edges_mpi(*problemData->readGrid, rank, block_width, size);
//        if (iteration == 0 && rank == 3) {
//            print_problem(*problemData, rank, block_width);
//            printf("\n");
//        }
        if (iteration % SOLUTION_REPORT_INTERVAL == 0 || iteration == NUM_SIMULATION_STEPS) {
            int sum = count_alive(*problemData, rank, block_width);
            int sum_prev;

            int des = rank + 1;
            int source = rank - 1;
            if (rank == 0)
                MPI_Send(&sum, 1, MPI_INT, des, 5, MPI_COMM_WORLD);
            else if (rank == size - 1) {
                MPI_Recv(&sum_prev, 1, MPI_INT, source, 5, MPI_COMM_WORLD, nullptr);
                sum += sum_prev;
                std::cout << "Iteration " << iteration << ": " << sum << " cells alive." << std::endl;
            } else {
                MPI_Recv(&sum_prev, 1, MPI_INT, source, 5, MPI_COMM_WORLD, nullptr);
                sum += sum_prev;
                MPI_Send(&sum, 1, MPI_INT, des, 5, MPI_COMM_WORLD);
            }
        }
        if (iteration == NUM_SIMULATION_STEPS)
            break;

        evolve(*problemData, rank, block_width);

        problemData->swapGrids();
    }

    delete problemData;
    if (rank == 0)
        std::cout << "DONE" << std::endl;
    MPI_Finalize();
    return 0;
}
