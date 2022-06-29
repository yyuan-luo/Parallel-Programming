#ifndef _LIFE_H_
#define _LIFE_H_

#define GRID_SIZE 1504
#define NUM_SIMULATION_STEPS 4111
#define SOLUTION_REPORT_INTERVAL 250

typedef bool Grid[GRID_SIZE][GRID_SIZE];

class ProblemData {
private:
    bool grid1[GRID_SIZE][GRID_SIZE]{};
    bool grid2[GRID_SIZE][GRID_SIZE]{};

public:
    bool(*readGrid)[GRID_SIZE][GRID_SIZE] = &grid1;
    bool(*writeGrid)[GRID_SIZE][GRID_SIZE] = &grid2;

    void swapGrids() {
        auto* tmp = readGrid;
        readGrid = writeGrid;
        writeGrid = tmp;
    }
};

#endif 
