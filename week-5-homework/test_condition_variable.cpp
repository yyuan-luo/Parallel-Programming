//
// Created by parallels on 6/17/22.
//

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

std::mutex m;
std::condition_variable cv;
int count = 0;
std::queue<int> queue;

void worker_thread()
{
    while (count) {
        // Wait until main() sends data
        std::unique_lock <std::mutex> lk(m);
        cv.wait(lk, [] { return count; });

        // after the wait, we own the lock.
        std::cout << "Worker thread is working\n";

        count--;

        // Send data back to main()
        std::cout << "count in thread is " << count << " and the data from the queue is " << queue.front() << "\n";
        queue.pop();
        // Manual unlocking is done before notifying, to avoid waking up
        // the waiting thread only to block again (see notify_one for details)
        lk.unlock();
    }
}

void distributor_thread() {
    std::cout << "distributor now starts to distribute the tasks\n";
    for (int i = 0; i < 100; ++i) {
        std::unique_lock<std::mutex> lk(m);
        queue.push(i);
        count++;
        lk.unlock();
        cv.notify_one();
    }
}

int main()
{
    std::thread distributor(distributor_thread);
    std::thread worker(worker_thread);
    std::thread worker1(worker_thread);
    std::thread worker2(worker_thread);
    std::thread worker3(worker_thread);
    std::thread worker4(worker_thread);

    distributor.join();
    worker.join();
    worker1.join();
    worker2.join();
    worker3.join();
    worker4.join();
}