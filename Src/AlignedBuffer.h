#pragma once

#include <stdint.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

class AlignedBuffer {
private:
    char* _data;
    size_t _size;

public:
    AlignedBuffer();
    AlignedBuffer(size_t size);
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    ~AlignedBuffer();
    AlignedBuffer& operator=(AlignedBuffer&& other);

    char* data();
    char const* data() const;
    size_t size() const;
};

template<typename T, size_t BS = 1024>
class Arena {
    static constexpr size_t block_size = BS;

    static_assert(std::is_trivial_v<T>);
    static_assert(1 <= block_size && block_size < 0x1000000);

private:
    struct Block {
        AlignedBuffer buffer;
        std::atomic<size_t> use_count;

        Block()
            : buffer(block_size * sizeof(T))
            , use_count(0)
        {
        }

        T* data() {
            return (T*)buffer.data();
        }

        T const* data() const {
            return (T const*)buffer.data();
        }

        T& operator[](size_t index) {
            return data()[index];
        }

        T const& operator[](size_t index) const {
            return data()[index];
        }

        //size_t increment_use_count(size_t delta) {
        //    return _use_count.fetch_add(delta, std::memory_order_release) + delta;
        //}
        //
        //size_t decrement_use_count(size_t delta) {
        //    return _use_count.fetch_sub(delta, std::memory_order_release) - delta;
        //}
        //
        //size_t use_count() const {
        //    _use_count.load(std::memory_order_acquire);
        //}
    };

    std::mutex _mutex;
    std::vector<Block*> _block_list;
    std::vector<size_t> _empty_block_index_list;

    void get_empty_block(Block*& block_ptr, size_t& block_index) {
        std::lock_guard guard(_mutex);
        if (_empty_block_index_list.empty()) {
            std::unique_ptr<Block> new_block_owning_ptr = std::make_unique<Block>();
            block_ptr = new_block_owning_ptr.get();
            block_index = _block_list.size();
            _block_list.push_back(new_block_owning_ptr.release());
        } else {
            block_index = _empty_block_index_list.back();
            _empty_block_index_list.pop_back();
            block_ptr = _block_list[block_index];
        }
    }

public:
    Arena() {
    }

    ~Arena() {
        for (Block* block_ptr : _block_list) {
            delete block_ptr;
        }
    }

    class Allocator {
    private:
        Arena* _arena_ptr;
        Block* _current_block_ptr;
        size_t _current_block_index;
        size_t _current_local_index;

    public:
        Allocator() {
            _arena_ptr = nullptr;
            _current_block_ptr = nullptr;
        }

        Allocator(Arena& arena) {
            _arena_ptr = &arena;
            _current_block_ptr = nullptr;
        }

        Allocator(Allocator&& other) {
            _arena_ptr = other._arena_ptr;
            _current_block_ptr = other._current_block_ptr;
            _current_block_index = other._current_block_index;
            _current_local_index = other._current_local_index;
            other._current_block_ptr = nullptr;
        }

        Allocator& operator=(Allocator&& other) {
            _arena_ptr = other._arena_ptr;
            _current_block_ptr = other._current_block_ptr;
            _current_block_index = other._current_block_index;
            _current_local_index = other._current_local_index;
            other._current_block_ptr = nullptr;
            return *this;
        }

        ~Allocator() {
        }

        T& alloc(uint64_t& composite_index) {
            if (!_current_block_ptr) {
                if (!_arena_ptr) {
                    throw std::runtime_error("Allocator is not assigned to an Arena");
                }
                _arena_ptr->get_empty_block(_current_block_ptr, _current_block_index);
                _current_local_index = 0;
            }
            composite_index = _current_block_index * block_size + _current_local_index;
            T& elem = (*_current_block_ptr)[_current_local_index];
            _current_block_ptr->use_count.fetch_add(1, std::memory_order_acquire);
            _current_local_index += 1;
            if (_current_local_index >= block_size) {
                _current_block_ptr = nullptr;
            }
            return elem;
        }
    };

    Allocator allocator() {
        return Allocator(*this);
    }

    class View {
    private:
        Arena* _arena_ptr;
        /*  A copy of _arena_ptr->_block_list, so that the view does not get invalidated
            if _arena_ptr->get_empty_block extends the list and causes a reallocation. */
        std::vector<Block*> _block_list;
        std::vector<size_t> _pending_decrements;

    public:
        View(Arena& arena) {
            _arena_ptr = &arena;
        }

        View(View const& other) {
            _arena_ptr = other._arena_ptr;
        }

        View(View&& other) {
            _arena_ptr = other._arena_ptr;
            _block_list = std::move(other._block_list);
            _pending_decrements = std::move(other._pending_decrements);
            other._pending_decrements.clear();
        }

        View& operator=(View const& other) {
            if (_arena_ptr != other._arena_ptr) {
                commit_decrements();
                _arena_ptr = other._arena_ptr;
                _block_list.clear();
                _pending_decrements.clear();
            }
            return *this;
        }

        View& operator=(View&& other) {
            if (_arena_ptr != other._arena_ptr) {
                commit_decrements();
                _arena_ptr = other._arena_ptr;
                _block_list = std::move(other._block_list);
                _pending_decrements = std::move(other._pending_decrements);
                other._pending_decrements.clear();
            }
            return *this;
        }

        ~View() {
            commit_decrements();
        }

        T& operator[](uint64_t composite_index) {
            size_t block_index = composite_index / block_size;
            size_t local_index = composite_index % block_size;
            if (block_index >= _block_list.size()) {
                std::lock_guard<std::mutex> guard(_arena_ptr->_mutex);
                _block_list = _arena_ptr->_block_list; // copy
                _pending_decrements.resize(_block_list.size(), 0);
            }
            if (block_index >= _block_list.size()) {
                throw std::runtime_error("invalid index for an Arena::View");
            }
            return (*_block_list[block_index])[local_index];
        }

        void schedule_decrement(uint64_t composite_index) {
            size_t block_index = composite_index / block_size;
            if (block_index >= _block_list.size()) {
                throw std::runtime_error("invalid index for an Arena::View");
            }
            _pending_decrements[block_index] += 1;
        }

        void commit_decrements() {
            std::unique_lock<std::mutex> guard(_arena_ptr->_mutex, std::defer_lock);
            for (size_t block_index = 0; block_index < _pending_decrements.size(); ++block_index) {
                size_t delta = _pending_decrements[block_index];
                if (delta != 0) {
                    size_t old_use_count = _block_list[block_index]->use_count.fetch_sub(delta, std::memory_order_release);
                    if (old_use_count == delta) {
                        if (!guard.owns_lock()) {
                            guard.lock();
                        }
                        _arena_ptr->_empty_block_index_list.push_back(block_index);
                    }
                    _pending_decrements[block_index] = 0;
                }
            }
        }
    };

    View view() {
        return View(*this);
    }
};
