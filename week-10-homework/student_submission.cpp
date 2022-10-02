#include <cstddef> // for size_t
#include "Utility.h"
#include <algorithm>
#include <mpi.h>

enum class Border : size_t {
    Top,
    Bottom,
    Left,
    Right
};

double local_min{0.0};
double local_max{0.0};

struct InformationExchangeData {
    std::pair<int, Border> neighbors[2];

    double *ghost_layers[4]{nullptr};

    double *strided_packed_message_left{nullptr};
    double *strided_packed_message_right{nullptr};
};

void init_information_exchange_data(InformationExchangeData &ied, ProblemData &pd, std::pair<int, Border> nbs[2]) {
    ied.neighbors[0] = nbs[0];
    ied.neighbors[1] = nbs[1];

    std::fill(ied.ghost_layers, ied.ghost_layers + 4, nullptr);
    for (auto &pr: ied.neighbors) {
        ied.ghost_layers[static_cast<size_t>(pr.second)] = new double[pd.dimension]{};
    }

    ied.strided_packed_message_left = new double[pd.dimension];
    ied.strided_packed_message_right = new double[pd.dimension];
}

void free_information_exchange_data(InformationExchangeData &ied) {
    delete[] ied.strided_packed_message_left;
    delete[] ied.strided_packed_message_right;

    for (int i = 0; i < 4; ++i) {
        delete[] ied.ghost_layers[i];
    }
}

void pack_strided_halo_to_continous(ProblemData &pd, InformationExchangeData &ied) {
    for (int y = 0; y < pd.dimension; ++y) {
        double **domain = Utility::get_domain(pd);
        ied.strided_packed_message_left[y] = domain[y][0];
        ied.strided_packed_message_right[y] = domain[y][pd.dimension - 1];
    }
}

void compute_stencil(ProblemData &pd, InformationExchangeData &ied) {
    // For evert cell with coordinates (y,x) compute the influx from neighbor cells
    // Apply reflecting boundary conditions
    local_max = std::numeric_limits<double>::min();
    local_min = std::numeric_limits<double>::max();

    double **domain = Utility::get_domain(pd);
    double **new_domain = Utility::get_new_domain(pd);

#pragma omp parallel for firstprivate(domain, new_domain) reduction(max : local_max) reduction(min : local_min)
    for (size_t y = 0; y < pd.dimension; y++)
    {
        for (size_t x = 0; x < pd.dimension; x++)
        {
            double cell_water = domain[y][x];
            double update = 0.0;

            // Add left neighbor
            if (x != 0)
            {
                double difference = domain[y][x - 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add right neighbor
            if (x != pd.dimension - 1)
            {
                double difference = domain[y][x + 1] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add lower neighbor
            if (y != 0)
            {
                double difference = domain[y - 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // Add upper neighbor
            if (y != pd.dimension - 1)
            {
                double difference = domain[y + 1][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            // If we are not in a true corner, we need to check the ghost layers and ask them for water height
            if (x == 0 && ied.ghost_layers[static_cast<size_t>(Border::Left)] != nullptr)
            {
                double difference = ied.ghost_layers[static_cast<size_t>(Border::Left)][y] - cell_water;
                update += difference / Utility::viscosity_factor;
            }
            if (x == pd.dimension - 1 && ied.ghost_layers[static_cast<size_t>(Border::Right)] != nullptr)
            {
                double difference = ied.ghost_layers[static_cast<size_t>(Border::Right)][y] - cell_water;
                update += difference / Utility::viscosity_factor;
            }
            if (y == pd.dimension - 1 && ied.ghost_layers[static_cast<size_t>(Border::Bottom)] != nullptr)
            {
                double difference = ied.ghost_layers[static_cast<size_t>(Border::Bottom)][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }
            if (y == 0 && ied.ghost_layers[static_cast<size_t>(Border::Top)] != nullptr)
            {
                double difference = ied.ghost_layers[static_cast<size_t>(Border::Top)][x] - cell_water;
                update += difference / Utility::viscosity_factor;
            }

            double waterheight = cell_water + update;

            if (waterheight > local_max)
            {
                local_max = waterheight;
            }
            else if (waterheight < local_min)
            {
                local_min = waterheight;
            }

            new_domain[y][x] = waterheight;
        }
    }
}

bool termination_criteria_fulfilled(ProblemData &pd) {
    double global_max = 0.0;
    double global_min = 0.0;

    // TODO @Students:
    // track min/max across ranks
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if ((global_max - global_min) < Utility::threshold) {
        return true;
    }

    return false;
}

void exchange_halo(ProblemData &pd, InformationExchangeData &ied) {
    // TODO @Students:
    // Implement halo exchange

    pack_strided_halo_to_continous(pd, ied);

    for (const auto &pr: ied.neighbors) {
        const int neighbor_rank = pr.first;
        const Border location = pr.second;
        int retcode = -1;

        double **domain = Utility::get_domain(pd);

        switch (location) {
            case Border::Top: {
                retcode = MPI_Sendrecv(domain[0], pd.dimension, MPI_DOUBLE, neighbor_rank,
                                       static_cast<size_t>(Border::Top),
                                       ied.ghost_layers[static_cast<size_t>(Border::Top)], pd.dimension, MPI_DOUBLE,
                                       neighbor_rank,
                                       static_cast<size_t>(Border::Bottom), MPI_COMM_WORLD, NULL);
                break;
            }
            case Border::Bottom: {
                retcode = MPI_Sendrecv(domain[pd.dimension - 1], pd.dimension, MPI_DOUBLE, neighbor_rank,
                                       static_cast<size_t>(Border::Bottom),
                                       ied.ghost_layers[static_cast<size_t>(Border::Bottom)], pd.dimension, MPI_DOUBLE,
                                       neighbor_rank, static_cast<size_t>(Border::Top),
                                       MPI_COMM_WORLD, NULL);
                break;
            }
            case Border::Left: {
                retcode = MPI_Sendrecv(ied.strided_packed_message_left, pd.dimension, MPI_DOUBLE, neighbor_rank,
                                       static_cast<size_t>(Border::Left),
                                       ied.ghost_layers[static_cast<size_t>(Border::Left)], pd.dimension, MPI_DOUBLE,
                                       neighbor_rank, static_cast<size_t>(Border::Right),
                                       MPI_COMM_WORLD, NULL);
                break;
            }
            case Border::Right: {
                retcode = MPI_Sendrecv(ied.strided_packed_message_right, pd.dimension, MPI_DOUBLE, neighbor_rank,
                                       static_cast<size_t>(Border::Right),
                                       ied.ghost_layers[static_cast<size_t>(Border::Right)], pd.dimension, MPI_DOUBLE,
                                       neighbor_rank, static_cast<size_t>(Border::Left),
                                       MPI_COMM_WORLD, NULL);
                break;
            }
            default: {
                throw std::runtime_error("Unhandled case");
            }
        }
        if (retcode != MPI_SUCCESS) {
            throw std::runtime_error(std::to_string(retcode));
        }
    }
}

unsigned long long simulate(ProblemData &pd, InformationExchangeData &ied, int rank) {
    volatile bool terminate_criteria_met = false;

    struct timespec start, end;
    while (!terminate_criteria_met) {
        exchange_halo(pd, ied);
        compute_stencil(pd, ied);
        terminate_criteria_met = termination_criteria_fulfilled(pd);
        Utility::switch_arrays(pd);
        pd.patch_updates += 1;
    }

    return pd.patch_updates;
}

double **generate_initial_water_height(ProblemData &pd, int seed) {
    Utility::generator.seed(seed);
    size_t half_dimension = pd.dimension / 2;
    size_t y_offsets[2] = {0, half_dimension};
    size_t x_offsets[2] = {0, half_dimension};

    constexpr
    size_t total_domain_count = 4;
    double **data = new double *[total_domain_count];

    for (size_t i = 0; i < total_domain_count; i++) {
        data[i] = new double[pd.dimension * pd.dimension];
    }

    size_t domain_counter = 0;
    for (size_t yoff: y_offsets) {
        for (size_t xoff: x_offsets) {
            for (size_t y = 0 + yoff; y < half_dimension + yoff; y++) {
                for (size_t x = 0 + xoff; x < half_dimension + xoff; x++) {
                    data[domain_counter][y * pd.dimension + x] = Utility::get_water_height(y, x);
                }
            }

            domain_counter += 1;
        }
    }

    return data;
}

int main(int argc, char **argv) {
    // TODO @Students:
    // Initialize MPI
    int provided_threading;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_threading);
    if (provided_threading < MPI_THREAD_FUNNELED) {
        printf("The threading support level is lesser than that demanded\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    ProblemData pd;
    InformationExchangeData ied;

    std::pair<int, Border> neighbors[2];
    switch (rank)
    {
        case (0):
        {
            neighbors[0] = std::make_pair(1, Border::Right);
            neighbors[1] = std::make_pair(2, Border::Bottom);
            break;
        }
        case (1):
        {
            neighbors[0] = std::make_pair(0, Border::Left);
            neighbors[1] = std::make_pair(3, Border::Bottom);
            break;
        }
        case (2):
        {
            neighbors[0] = std::make_pair(0, Border::Top);
            neighbors[1] = std::make_pair(3, Border::Right);
            break;
        }
        case (3):
        {
            neighbors[0] = std::make_pair(1, Border::Top);
            neighbors[1] = std::make_pair(2, Border::Left);
            break;
        }
    }
    // TODO @Students:
    // Think about minimizing the array size at each rank. 
    // this might require additional changes elsewhere
    Utility::init_problem_data(pd, Utility::domain_size / 2);
    init_information_exchange_data(ied, pd, neighbors);

    size_t packed_arr_len = pd.dimension * pd.dimension;

    double *data_received = nullptr;
    double **initial_water_heights = nullptr;

    if (rank == 0)
    {
        int seed;
        Utility::readInput(seed);
        initial_water_heights = generate_initial_water_height(pd, seed);
        MPI_Send(initial_water_heights[3], packed_arr_len, MPI_DOUBLE, 3, 0, MPI_COMM_WORLD);
        MPI_Send(initial_water_heights[2], packed_arr_len, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
        MPI_Send(initial_water_heights[1], packed_arr_len, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        data_received = initial_water_heights[0];
    }
    else
    {
        data_received = new double[packed_arr_len];
        MPI_Recv(data_received, packed_arr_len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // TODO @Students:
    // Initialize MPI find a way to send the initial data to the domains of other MPI Ranks

    Utility::apply_initial_water_height(pd, data_received);

    delete[] data_received;

    if (rank == 0)
    {
        for (size_t i = 1; i < 4; i++)
        {
            delete[] initial_water_heights[i];
        }
    }

    delete[] initial_water_heights;

    simulate(pd, ied, rank);

    if (rank == 0)
    {
        std::cout << pd.patch_updates << std::endl;
    }

    Utility::free_problem_data(pd);
    free_information_exchange_data(ied);

    std::cout << pd.patch_updates << std::endl;

    // TODO @Students:
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
