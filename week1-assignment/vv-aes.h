//
// Created by Vincent Bode on 21/04/2020.
// Updated by Dennis-Florian Herr 28/04/2022.
//

#ifndef ASSIGNMENTS_VV_AES_H
#define ASSIGNMENTS_VV_AES_H

#include <limits>

#define MESSAGE_BLOCK_SIZE 6
#define UNIQUE_CHARACTERS 256
#define NUM_ROUNDS 9
#define NUM_ITERATIONS 400000

/*
 * This is the message given to you to encrypt to verify the correctness and speed of your approach. Unfortunately,
 * there are no state secrets to be found here. Those would have probably made a pretty penny on the black market.
 */
uint8_t message[MESSAGE_BLOCK_SIZE][MESSAGE_BLOCK_SIZE] = {
        {'T', 'H', 'I', 'S', ' ', 'I'},
        {'S', ' ', 'A', 'V', 'E', 'R'},
        {'Y', ' ', 'S', 'E', 'C', 'R'},
        {'E', 'T', ' ', 'M', 'E', 'S'},
        {'S', 'A', 'G', 'E', '!', ' '},
        {'O', ' ', 'R', 'L', 'Y', '?'}
};

/*
 * The set of all keys generated at runtime and the index of the current key.
 */
int currentKey = 0;
uint8_t allKeys[NUM_ROUNDS][MESSAGE_BLOCK_SIZE][MESSAGE_BLOCK_SIZE];

/*
 * Use this like a 2D-Matrix key[MESSAGE_BLOCK_SIZE][MESSAGE_BLOCK_SIZE];
 * The key is only handed to you when it's time to actually encrypt something.
 */
uint8_t (*key)[MESSAGE_BLOCK_SIZE];


/*
 * This is the source list of characters. If you wish to translate a character, find its index in this list. The
 * corresponding output character resides at the same index in the substituted character list.
 */
uint8_t originalCharacter[] = { 8, 100, 167, 27, 45, 120, 24, 43, 222, 133, 86, 131, 181, 15, 203, 116, 226, 158, 205, 87, 21, 95, 250, 216, 16, 22, 239, 254, 84, 88, 60, 233, 214, 208, 187, 225, 69, 202, 145, 137, 164, 121, 125, 242, 89, 97, 65, 36, 51, 231, 111, 243, 114, 130, 229, 98, 228, 169, 132, 9, 135, 141, 49, 236, 48, 157, 175, 223, 61, 219, 42, 14, 140, 209, 142, 73, 55, 83, 247, 109, 6, 168, 76, 115, 112, 238, 44, 150, 151, 166, 154, 193, 47, 99, 170, 232, 155, 2, 77, 144, 119, 196, 106, 58, 38, 163, 235, 156, 80, 190, 227, 186, 189, 37, 204, 59, 67, 28, 19, 10, 105, 63, 3, 127, 162, 206, 18, 113, 136, 180, 81, 124, 244, 50, 74, 128, 212, 251, 179, 4, 52, 35, 23, 54, 94, 26, 129, 221, 39, 198, 207, 66, 5, 1, 252, 147, 152, 234, 92, 148, 192, 40, 195, 78, 200, 161, 93, 46, 211, 31, 188, 62, 201, 176, 68, 139, 33, 30, 56, 17, 241, 183, 172, 220, 57, 75, 149, 213, 25, 255, 215, 177, 159, 143, 246, 12, 0, 253, 110, 102, 245, 174, 171, 178, 210, 160, 173, 11, 118, 64, 122, 29, 126, 249, 185, 182, 101, 194, 138, 41, 72, 96, 199, 237, 123, 104, 107, 82, 146, 248, 230, 70, 153, 134, 165, 34, 90, 85, 13, 108, 91, 184, 191, 197, 32, 117, 240, 7, 20, 218, 217, 103, 224, 53, 71, 79 };
/*
 * This is the output list of characters. If you wish to translate a character, find its index in the original list.
 * The corresponding output character resides at the same index in this list.
 */
uint8_t substitutedCharacter[] = { 89, 35, 25, 135, 107, 68, 199, 208, 90, 209, 211, 34, 97, 78, 245, 26, 96, 9, 189, 123, 33, 36, 22, 177, 115, 81, 167, 11, 129, 254, 148, 203, 216, 223, 126, 29, 198, 87, 77, 76, 52, 190, 161, 156, 61, 207, 143, 58, 73, 117, 184, 24, 4, 56, 10, 200, 12, 114, 1, 53, 125, 233, 80, 253, 252, 215, 168, 41, 138, 159, 50, 186, 137, 45, 27, 222, 46, 175, 158, 244, 195, 172, 238, 109, 51, 241, 49, 154, 155, 101, 38, 150, 60, 250, 84, 153, 201, 17, 64, 144, 131, 183, 171, 55, 170, 72, 127, 91, 65, 111, 20, 239, 179, 18, 39, 16, 85, 182, 94, 99, 7, 235, 212, 237, 92, 100, 248, 122, 146, 243, 221, 30, 141, 140, 220, 230, 218, 112, 232, 210, 28, 191, 124, 217, 174, 128, 132, 6, 86, 74, 31, 247, 157, 142, 79, 231, 213, 136, 44, 23, 240, 70, 246, 108, 224, 57, 88, 196, 178, 193, 62, 180, 104, 165, 229, 116, 194, 75, 242, 71, 149, 15, 164, 214, 169, 185, 160, 121, 3, 66, 21, 204, 228, 227, 139, 219, 176, 48, 130, 225, 145, 32, 163, 42, 197, 63, 59, 43, 113, 106, 120, 152, 19, 8, 13, 133, 102, 47, 82, 147, 166, 40, 98, 134, 67, 226, 251, 236, 69, 118, 93, 206, 151, 187, 173, 110, 14, 255, 119, 54, 202, 103, 5, 2, 83, 0, 37, 95, 105, 192, 188, 234, 162, 249, 181, 205 };
uint8_t polynomialCoefficients[MESSAGE_BLOCK_SIZE][MESSAGE_BLOCK_SIZE] = {
        { 3, 1, 6, 5, 9, 4},
        { 9, 6, 3, 8, 5, 2},
        { 1, 2, 3, 4, 5, 6},
        { 9, 8, 7, 6, 5, 4},
        { 3, 8, 6, 5, 9, 1},
        { 3, 5, 2, 8, 7, 9}
};

/*
 * Generate some keys that can be used for encryption based on the seed set from the input.
 */
void generate_keys() {
    // Fill the key block
    for(auto & currentKey : allKeys) {
        for (auto & row : currentKey) {
            for (unsigned char & column : row) {
                column = rand() % std::numeric_limits<uint8_t>::max();
            }
        }
    }
}

void readInput() {
    std::cout << "READY" << std::endl;
    unsigned int seed = 0;
    std::cin >> seed;

    std::cerr << "Using seed " << seed << std::endl;
    if(seed == 0) {
        std::cerr << "Warning: default value 0 used as seed." << std::endl;
    }

    // Set the pseudo random number generator seed used for generating encryption keys
    srand(seed);

    generate_keys();
}

void writeOutput() {
    // Output the current state of the message in hexadecimal.
    for (int row = 0; row < MESSAGE_BLOCK_SIZE; row++) {
        std::cout << std::hex << (int) message[row][0] << (int) message[row][1] << (int) message[row][2]
                  << (int) message[row][3];
    }
    // This stops the timer.
    std::cout << std::endl << "DONE" << std::endl;
}

/*
 * This is a utility method. It determines the next key to use from the set of pre-generated keys. In a real
 * cryptographic system, the subsequent keys are generated from a key schedule from the original key. To keep the code
 * short, we have omitted this behavior.
 */
void set_next_key() {
    key = &allKeys[currentKey][0];
    currentKey = (currentKey + 1) % NUM_ROUNDS;
}

#endif //ASSIGNMENTS_VV_AES_H
