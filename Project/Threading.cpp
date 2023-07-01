#include "Threading.h"
#include <chrono>

constexpr bool restrict_to_one_worker = false;

using namespace std::chrono_literals;

thread_local WorkerId current_worker_id_value = WorkerId(0);

ProcessorControl::Guard::Guard(ProcessorControl* control_ptr) {
    _control_ptr = control_ptr;
}

ProcessorControl::Guard::Guard(Guard&& other) {
    _control_ptr = other._control_ptr;
    other._control_ptr = nullptr;
}

ProcessorControl::Guard& ProcessorControl::Guard::operator=(Guard&& other) {
    release();
    _control_ptr = other._control_ptr;
    other._control_ptr = nullptr;
    return *this;
}

ProcessorControl::Guard::~Guard() {
    release();
}

ProcessorControl::Guard::operator bool() const {
    return (bool)_control_ptr;
}

void ProcessorControl::Guard::release() {
    if (_control_ptr) {
        _control_ptr->usage_end();
        _control_ptr = nullptr;
    }
}

bool ProcessorControl::is_live() {
    return _is_live.load(std::memory_order_relaxed);
}

void ProcessorControl::set_dead() {
    _is_live.store(false, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acq_rel);
    while (_usage_count.load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
    }
}

bool ProcessorControl::try_usage_start() {
    _usage_count.fetch_add(1, std::memory_order_acquire);
    if (_is_live.load(std::memory_order_relaxed)) {
        return true;
    } else {
        _usage_count.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }
}

void ProcessorControl::usage_end() {
    _usage_count.fetch_sub(1, std::memory_order_release);
}

ProcessorControl::Guard ProcessorControl::try_use() {
    if (try_usage_start()) {
        return Guard(this);
    } else {
        return Guard(nullptr);
    }
}

Processor::Processor(std::shared_ptr<ProcessorControl> control_ptr) {
    if (!control_ptr) {
        throw std::runtime_error("control_ptr is null");
    }
    _control_ptr = std::move(control_ptr);
}

bool Processor::is_live() {
    return _control_ptr->is_live();
}

bool Processor::has_pending_work() {
    if (auto guard = _control_ptr->try_use()) {
        return has_pending_work_impl();
    } else {
        return false;
    }
}

bool Processor::work() {
    if (auto guard = _control_ptr->try_use()) {
        return work_impl();
    } else {
        return false;
    }
}

Scheduler::Worker::Worker(WorkerId id, Scheduler& scheduler)
    : _scheduler(scheduler)
{
    _idle_counter = 0;
    _thread = std::thread(&Worker::main, this, id);
}

Scheduler::Worker::~Worker() {
    _thread.join();
}

void Scheduler::Worker::iterate() {
    auto cache_prev = _processor_cache.before_begin();
    while (true) {
        auto cache_current = cache_prev;
        ++cache_current;
        if (cache_current == _processor_cache.end()) {
            break;
        }
        bool work_performed = (*cache_current)->work();
        if (work_performed) {
            std::shared_ptr<Processor> source_ptr = std::move(*cache_current);
            _processor_cache.erase_after(cache_prev);
            _processor_cache.push_front(std::move(source_ptr));
            _idle_counter = 0;
            return;
        }
        bool is_live = (*cache_current)->is_live();
        if (is_live) {
            cache_prev = cache_current;
        } else {
            _processor_cache_set.erase((*cache_current).get());
            _processor_cache.erase_after(cache_prev);
        }
    }

    {
        std::lock_guard<std::mutex> guard(_scheduler._mutex);
        auto sources_it = _scheduler._processors.begin();
        while (sources_it != _scheduler._processors.end()) {
            if (_processor_cache_set.count((*sources_it).get()) == 0) {
                if ((*sources_it)->has_pending_work()) {
                    std::shared_ptr<Processor> source_ptr = std::move(*sources_it);
                    _scheduler._processors.erase(sources_it);
                    _scheduler._processors.push_back(source_ptr);
                    _processor_cache_set.emplace(source_ptr.get());
                    _processor_cache.push_front(std::move(source_ptr));
                    return;
                } else {
                    if ((*sources_it)->is_live()) {
                        ++sources_it;
                    } else {
                        sources_it = _scheduler._processors.erase(sources_it);
                    }
                }
            } else {
                ++sources_it;
            }
        }
    }

    _idle_counter += 1;
    if (_idle_counter < 20) {
        std::this_thread::yield();
    } else if (_idle_counter < 50) {
        std::this_thread::sleep_for(1s);
    } else {
        std::this_thread::sleep_for(2s);
    }
}

void Scheduler::Worker::main(WorkerId id) {
    current_worker_id_value = id;
    while (!_scheduler._terminate_flag.load(std::memory_order_relaxed)) {
        iterate();
    }
}

Scheduler::Scheduler() {
    _terminate_flag.store(false, std::memory_order_relaxed);
    int worker_count = (std::thread::hardware_concurrency() * 3 + 3) / 4;
    if (restrict_to_one_worker) {
        worker_count = 1;
    }
    _workers.resize(worker_count);
    for (int i = 0; i < worker_count; ++i) {
        _workers[i] = std::make_unique<Worker>(WorkerId(i + 1), *this);
    }
    current_worker_id_value = WorkerId(worker_count + 1);
}

Scheduler::~Scheduler() {
    _terminate_flag.store(true, std::memory_order_relaxed);
}

Scheduler& Scheduler::instance() {
    static Scheduler singleton_instance;
    return singleton_instance;
}

void Scheduler::register_processor(std::shared_ptr<Processor> processor_ptr) {
    if (!processor_ptr) {
        throw std::runtime_error("processor_ptr is null");
    }
    Scheduler& self = instance();
    std::lock_guard<std::mutex> guard(self._mutex);
    self._processors.push_back(std::move(processor_ptr));
}

int Scheduler::total_worker_count() {
    return (int)(instance()._workers.size() + 1);
}

WorkerId Scheduler::current_worker_id() {
    return current_worker_id_value;
}
