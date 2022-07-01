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

#define CODE_VERSION 1

/*
  Apply the game of life rules on a Torus --> grid contains shadow rows and columns
  to simplify application of rules i.e. grid actually ranges from grid [ 1.. height - 2 ][ 1 .. width - 2]
*/
void evolve(ProblemData &problemData, int rank, int block_width, int sqrt_size) {
    auto &grid = *problemData.readGrid;
    auto &writeGrid = *problemData.writeGrid;
    // TODO: MPI_Send and MPI_Recv to be implemented
    // For each cell
    for (int x = max(1, (rank % block_num_horizontal) * block_width);
         x < min(GRID_SIZE - 1, (rank % block_num_horizontal + 1) * block_width); x++) {
        for (int y = max(1, (rank / block_num_horizontal) * block_width);
             y < min(GRID_SIZE - 1, (rank / block_num_horizontal + 1) * block_width); y++) {
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
    // TODO: change range based on rank
    for (int x = max(1, (rank % block_num_horizontal) * block_width);
         x < min(GRID_SIZE - 1, (rank % block_num_horizontal + 1) * block_width); x++) {
        for (int y = max(1, (rank / block_num_horizontal) * block_width);
             y < min(GRID_SIZE - 1, (rank / block_num_horizontal + 1) * block_width); y++) {
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

    int sqrt_size = sqrt(size)
    int block_width = GRID_SIZE / sqrt_size;

    int local_sum = 0, global_sum = 0;
    auto *problemData = new ProblemData;

    copy_edges(*problemData->readGrid);

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
            local_sum = count_alive(problemData, rank, block_width, sqrt_size);
        }

        evolve(*problemData);

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

