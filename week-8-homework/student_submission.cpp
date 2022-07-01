//
// Created by Vincent Bode on 08/07/2020.
//

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "life.h"
#include "Utility.h"
#include "VideoOutput.h"
#include <mpi.h>
#include <math.h>
#include <iostream>

#define CODE_VERSION 1

/*
  Apply the game of life rules on a Torus --> grid contains shadow rows and columns
  to simplify application of rules i.e. grid actually ranges from grid [ 1.. height - 2 ][ 1 .. width - 2]
*/
void evolve(ProblemData &problemData, int rank, int block_width, int sqrt_size) {
    auto &grid = *problemData.readGrid;
    auto &writeGrid = *problemData.writeGrid;
    // TODO: MPI_Send and MPI_Recv to be implemented
    if (rank == 0) {
        // top-left corner
        MPI_Send(&grid[1][1], 1, MPI_CXX_BOOL, 3, 0, MPI_COMM_WORLD);
        MPI_Recv(&grid[0][0], 1, MPI_CXX_BOOL, 3, 0, MPI_COMM_WORLD, nullptr);

        // top
        MPI_Send(&grid[1][1], block_width - 1, MPI_CXX_BOOL, 2, 1, MPI_COMM_WORLD);
        MPI_Recv(&grid[0][1], block_width - 1, MPI_CXX_BOOL, 2, 1, MPI_COMM_WORLD, nullptr);

        // left
        bool tmp[block_width - 1];
        for (int i = 1; i < block_width; ++i)
            tmp[i - 1] = grid[i][1];
        MPI_Send(tmp, block_width - 1, MPI_CXX_BOOL, 1, 2, MPI_COMM_WORLD);
        MPI_Recv(tmp, block_width - 1, MPI_CXX_BOOL, 1, 2, MPI_COMM_WORLD, nullptr);
        for (int i = 1; i < block_width; ++i)
            grid[i][0] = tmp[i - 1];

        // right
        for (int i = 1; i < block_width; ++i) {
            tmp[i - 1] = grid[i][block_width - 1];
        }
        MPI_Send(&tmp, block_width - 1, MPI_CXX_BOOL, 1, 3, MPI_COMM_WORLD);
        MPI_Recv(tmp, block_width - 1, MPI_CXX_BOOL, 1, 3, MPI_COMM_WORLD, nullptr);
        for (int i = 1; i < block_width; ++i) {
            grid[block_width][i] = tmp[i - 1];
        }

        // bottom
        MPI_Send(&grid[block_width - 1][1], block_width - 1, MPI_CXX_BOOL, 2, 4, MPI_COMM_WORLD);
        MPI_Recv(&grid[block_width][1], block_width - 1, MPI_CXX_BOOL, 2, 4, MPI_COMM_WORLD, nullptr);

    } else if (rank == 1) {
        MPI_Send(&grid[1][GRID_SIZE - 2], 1, MPI_CXX_BOOL, 2, 0, MPI_COMM_WORLD);
#if DEBUG
        std::cout << "rank " << rank << " sent " << grid[1][GRID_SIZE - 2] << std::endl;
#endif

        MPI_Recv(&grid[0][GRID_SIZE - 1], 1, MPI_CXX_BOOL, 2, 1, MPI_COMM_WORLD, nullptr);
#if DEBUG
        std::cout << "rank " << rank << " received " << grid[0][GRID_SIZE - 1] << std::endl;
#endif
        bool tmp[GRID_SIZE - 1];
        for (int i = 1; i < GRID_SIZE; ++i) {
            tmp[i - 1] = grid[GRID_SIZE - 2][i];
        }
        MPI_Send(&tmp, block_width - 1, MPI_CXX_BOOL, 0, 3, MPI_COMM_WORLD);


        MPI_Send(&grid[0][GRID_SIZE], block_width - 1, MPI_CXX_BOOL, 4, MPI_COMM_WORLD);


    } else if (rank == 2) {
        MPI_Recv(&grid[GRID_SIZE - 1][0], 1, MPI_CXX_BOOL, 1, 0, MPI_COMM_WORLD, nullptr);
#if DEBUG
        std::cout << "rank " << rank << " sent " << grid[GRID_SIZE - 1][0] << std::endl;
#endif

        MPI_Send(&grid[GRID_SIZE - 2][1], 1, MPI_CXX_BOOL, 1, 1, MPI_COMM_WORLD);
#if DEBUG
        std::cout << "rank " << rank << " received " << grid[GRID_SIZE - 2][1] << std::endl;
#endif
    } else if (rank == 3) {
        MPI_Recv(&grid[GRID_SIZE - 1][GRID_SIZE - 1], 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, nullptr);
#if DEBUG
        std::cout << "rank " << rank << " sent " << grid[GRID_SIZE - 1][GRID_SIZE - 1] << std::endl;
#endif

        MPI_Send(&grid[GRID_SIZE - 2][GRID_SIZE - 2], 1, MPI_CXX_BOOL, 0, 1, MPI_COMM_WORLD);
#if DEBUG
        std::cout << "rank " << rank << " received " << grid[GRID_SIZE - 2][GRID_SIZE - 2] << std::endl;
#endif
    }
    // For each cell
    for (int i = std::max(1, (rank % sqrt_size) * block_width);
         i < std::min(GRID_SIZE - 1, (rank % sqrt_size + 1) * block_width); i++) {
        for (int j = std::max(1, (rank / sqrt_size) * block_width);
             j < std::min(GRID_SIZE - 1, (rank / sqrt_size + 1) * block_width); j++) {
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
void copy_edges(bool(&grid)[GRID_SIZE][GRID_SIZE]) {
    // Copy data to the boundaries
    for (int i = 1; i < GRID_SIZE - 1; i++) {
        // join rows together
        grid[i][0] = grid[i][GRID_SIZE - 2];
        grid[i][GRID_SIZE - 1] = grid[i][1];
    }

    for (int j = 1; j < GRID_SIZE - 1; j++) {
        // join columns together
        grid[0][j] = grid[GRID_SIZE - 2][j];
        grid[GRID_SIZE - 1][j] = grid[1][j];
    }

    // Fix corners
    grid[0][0] = grid[GRID_SIZE - 2][GRID_SIZE - 2];
    grid[GRID_SIZE - 1][GRID_SIZE - 1] = grid[1][1];
    grid[0][GRID_SIZE - 1] = grid[GRID_SIZE - 2][1];
    grid[GRID_SIZE - 1][0] = grid[1][GRID_SIZE - 2];
}

int count_alive(ProblemData &data, int rank, int block_width, int sqrt_size) {
    auto &grid = *data.readGrid;
    int counter = 0;
    for (int x = std::max(1, (rank % sqrt_size) * block_width);
         x < std::min(GRID_SIZE - 1, (rank % sqrt_size + 1) * block_width); x++) {
        for (int y = std::max(1, (rank / sqrt_size) * block_width);
             y < std::min(GRID_SIZE - 1, (rank / sqrt_size + 1) * block_width); y++) {
            if (grid[x][y]) {
                counter++;
            }
        }
    }
    return counter;
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
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "size: " << size << " rank: " << rank << std::endl;

    int sqrt_size = sqrt(size);
    int block_width = GRID_SIZE / sqrt_size;

    int local_sum = 0, global_sum = 0;
    auto *problemData = new ProblemData;

//    copy_edges(*problemData->readGrid);

    // As with Jack Sparrow's exercise, this needs FFMPEG (new and improved: this now works with more video players).
    // As an alternative, you can write individual png files to take a look at the data.
    if (activateVideoOutput) {
        VideoOutput::beginVideoOutput();
        VideoOutput::saveToFile(*problemData->readGrid, "grid_beginning.png");
    }

    Utility::readProblemFromInput(CODE_VERSION, *problemData);

    //TODO@Students: This is the main simulation. Parallelize it using MPI.
    for (int iteration = 0; iteration < NUM_SIMULATION_STEPS; ++iteration) {

        if (iteration % SOLUTION_REPORT_INTERVAL == 0) {
            // TODO: how to make sure when try to printf, every process stays at SOLUTION_REPORT_INTERVAL
            //  or that you can get the count info for this interval

            /** might be possible to send the local _sum info to one process, since MPI_send is blocking
             * or apply MPI_Isend(Non-blocking) to store this info of each process in one process and printed
             * by that process */
            local_sum = count_alive(*problemData, rank, block_width, sqrt_size);
        }

        evolve(*problemData, rank, block_width, sqrt_size);

        problemData->swapGrids();
    }

    Utility::outputSolution(*problemData);

    if (activateVideoOutput) {
        VideoOutput::endVideoOutput();
        VideoOutput::saveToFile(*problemData->readGrid, "grid_final.png");
    }

    delete problemData;
    MPI_Finalize();
    return 0;
}

