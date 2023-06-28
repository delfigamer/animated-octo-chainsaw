#pragma once

#include <atomic>
#include <forward_list>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class WorkerId: int;

class ProcessorControl {
private:
    std::atomic<int> _usage_count = 0;
    std::atomic<bool> _is_live = true;

public:
    class Guard {
    private:
        ProcessorControl* _control_ptr;
        Guard(ProcessorControl* control_ptr);
    public:
        friend class ProcessorControl;

        Guard(Guard&& other);
        Guard& operator=(Guard&& other);
        ~Guard();

        explicit operator bool() const;
        void release();
    };

    bool is_live();
    void set_dead();
    bool try_usage_start();
    void usage_end();
    Guard try_use();
};

class Processor {
private:
    std::shared_ptr<ProcessorControl> _control_ptr;

public:
    Processor(std::shared_ptr<ProcessorControl> control_ptr);
    virtual ~Processor() = default;

    bool is_live();
    bool has_pending_work();
    bool work();
    virtual bool has_pending_work_impl() = 0;
    virtual bool work_impl() = 0;
};

class Scheduler {
    struct Worker {
        Scheduler& _scheduler;
        std::thread _thread;
        std::forward_list<std::shared_ptr<Processor>> _processor_cache;
        std::unordered_set<Processor*> _processor_cache_set;
        unsigned _idle_counter;

        Worker(WorkerId id, Scheduler& scheduler);
        ~Worker();

        void iterate();
        void main(WorkerId id);
    };

    std::atomic<bool> _terminate_flag;
    std::mutex _mutex;
    std::list<std::shared_ptr<Processor>> _processors;
    std::vector<std::unique_ptr<Worker>> _workers;

    Scheduler();
    ~Scheduler();

public:
    static Scheduler& instance();
    static void register_processor(std::shared_ptr<Processor> processor_ptr);
    static int total_worker_count();
    static WorkerId current_worker_id();
};
