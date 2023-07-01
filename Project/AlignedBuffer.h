#pragma once

#include <stdint.h>
#include <atomic>
#include <forward_list>
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
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    char* data();
    char const* data() const;
    size_t size() const;
    explicit operator bool() const;
};

struct iref {
    int& ir;
    iref(int& ir)
        : ir(ir)
    {
    }
};

template<typename T, size_t BS = 1024>
class Arena {
private:
    static constexpr size_t block_capacity = BS;
    static constexpr size_t block_byte_size = block_capacity * sizeof(T);
    
    static_assert(std::is_trivially_destructible_v<T>);
    static_assert(1 <= block_capacity && block_capacity < 0x1000000);

    std::mutex _mutex;
    std::vector<AlignedBuffer> _block_list;
    std::vector<std::pair<size_t, size_t>> _partial_blocks;
    size_t _used_block_count;
    
    void get_free_block(T*& block_data_ptr, size_t& block_index, size_t& next_local_index) {
        std::lock_guard guard(_mutex);
        if (!_partial_blocks.empty()) {
            std::pair<size_t, size_t> partial = _partial_blocks.back();
            block_index = partial.first;
            block_data_ptr = (T*)_block_list[block_index].data();
            next_local_index = partial.second;
            _partial_blocks.pop_back();
        } else if (_used_block_count < _block_list.size()) {
            block_index = _used_block_count;
            block_data_ptr = (T*)_block_list[block_index].data();
            next_local_index = 0;
            _used_block_count = block_index + 1;
        } else {
            block_index = _block_list.size();
            block_data_ptr = (T*)_block_list.emplace_back(block_byte_size).data();
            next_local_index = 0;
            _used_block_count = block_index + 1;
        }
    }
    
    void store_partial_block(size_t block_index, size_t next_local_index) {
        std::lock_guard guard(_mutex);
        _partial_blocks.emplace_back(block_index, next_local_index);
    }
    
public:
    Arena() {
    }

    ~Arena() {
    }

    void clear() {
        std::lock_guard guard(_mutex);
        _partial_blocks.clear();
        _used_block_count = 0;
    }

    class Allocator {
    private:
        Arena* _arena_ptr;
        T* _current_block_data_ptr;
        size_t _current_block_index;
        size_t _current_local_index;
    
    public:
        Allocator() {
            _arena_ptr = nullptr;
            _current_block_data_ptr = nullptr;
        }
    
        Allocator(Arena& arena) {
            _arena_ptr = &arena;
            _current_block_data_ptr = nullptr;
        }
    
        Allocator(Allocator&& other) {
            _arena_ptr = other._arena_ptr;
            _current_block_data_ptr = other._current_block_data_ptr;
            _current_block_index = other._current_block_index;
            _current_local_index = other._current_local_index;
            other._current_block_data_ptr = nullptr;
        }
    
        Allocator& operator=(Allocator&& other) {
            _arena_ptr = other._arena_ptr;
            _current_block_data_ptr = other._current_block_data_ptr;
            _current_block_index = other._current_block_index;
            _current_local_index = other._current_local_index;
            other._current_block_data_ptr = nullptr;
            return *this;
        }
    
        ~Allocator() {
            if (_current_block_data_ptr && _current_local_index < block_capacity) {
                _arena_ptr->store_partial_block(_current_block_index, _current_local_index);
            }
        }
    
        template<typename... As>
        T& emplace(uint64_t& composite_index, As&&... args) {
            if (!_current_block_data_ptr) {
                if (!_arena_ptr) {
                    throw std::runtime_error("Allocator is not assigned to an Arena");
                }
                _arena_ptr->get_free_block(_current_block_data_ptr, _current_block_index, _current_local_index);
            }
            composite_index = _current_block_index * block_capacity + _current_local_index;
            T* elem_ptr = _current_block_data_ptr + _current_local_index;
            new (elem_ptr) T(std::forward<As>(args)...);
            _current_local_index += 1;
            if (_current_local_index >= block_capacity) {
                _current_block_data_ptr = nullptr;
            }
            return *elem_ptr;
        }

        explicit operator bool() const {
            return _arena_ptr != nullptr;
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
        std::vector<T*> _block_data_ptr_list;
    
    public:
        View() {
            _arena_ptr = nullptr;
        }

        View(Arena& arena) {
            _arena_ptr = &arena;
        }
    
        View(View const& other) {
            _arena_ptr = other._arena_ptr;
        }
    
        View(View&& other) {
            _arena_ptr = other._arena_ptr;
            _block_data_ptr_list = std::move(other._block_data_ptr_list);
        }
    
        View& operator=(View const& other) {
            if (_arena_ptr != other._arena_ptr) {
                _arena_ptr = other._arena_ptr;
                _block_data_ptr_list.clear();
            }
            return *this;
        }
    
        View& operator=(View&& other) {
            if (_arena_ptr != other._arena_ptr) {
                _arena_ptr = other._arena_ptr;
                _block_data_ptr_list = std::move(other._block_data_ptr_list);
            }
            return *this;
        }
    
        ~View() {
        }
    
        void ensure_block_present(size_t block_index) {
            if (block_index >= _block_data_ptr_list.size()) {
                std::lock_guard<std::mutex> guard(_arena_ptr->_mutex);
                size_t old_size = _block_data_ptr_list.size();
                _block_data_ptr_list.resize(_arena_ptr->_block_list.size());
                for (size_t i = old_size; i < _block_data_ptr_list.size(); ++i) {
                    _block_data_ptr_list[i] = (T*)_arena_ptr->_block_list[i].data();
                }
                if (block_index >= _block_data_ptr_list.size()) {
                    throw std::runtime_error("invalid index for an Arena::View");
                }
            }
        }
    
        T& operator[](uint64_t composite_index) {
            size_t block_index = composite_index / block_capacity;
            size_t local_index = composite_index % block_capacity;
            ensure_block_present(block_index);
            return _block_data_ptr_list[block_index][local_index];
        }

        explicit operator bool() const {
            return _arena_ptr != nullptr;
        }
    };
    
    View view() {
        return View(*this);
    }
};
