// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/autograd/profiler.h>

#include <cerrno>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace deepmd::ptexpt {

// One finite Kineto window per backend instance, on the inference thread.
// Call counters describe outer backend computations, not LAMMPS timesteps.
// Never run this alongside another profiler on the same thread.
class WindowProfiler {
 public:
  static std::unique_ptr<WindowProfiler> from_environment(
      bool gpu, int gpu_id, std::function<void()> synchronize) {
    const char* stem = std::getenv("DP_PROFILER");
    if (!stem || !*stem) {
      return nullptr;
    }
    std::string file(stem);
    if (gpu) {
      file += "_gpu" + std::to_string(gpu_id);
    }
    file += ".json";
    return std::unique_ptr<WindowProfiler>(new WindowProfiler(
        file, read_count("DP_PROFILER_SKIP", 5, true),
        read_count("DP_PROFILER_ACTIVE", 10, false), gpu,
        std::move(synchronize)));
  }

  class Scope {
   public:
    explicit Scope(WindowProfiler* profiler)
        : profiler_(profiler), exceptions_(0) {
      if (profiler_ && !profiler_->finished_) {
        exceptions_ = std::uncaught_exceptions();
        profiler_->begin();
      } else {
        profiler_ = nullptr;
      }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    ~Scope() noexcept(false) {
      if (!profiler_) {
        return;
      }
      if (std::uncaught_exceptions() > exceptions_) {
        try {
          profiler_->end(true);
        } catch (const std::exception& error) {
          std::cerr << "PTExpt profiler cleanup failed: " << error.what()
                    << std::endl;
        }
      } else {
        profiler_->end(false);
      }
    }

   private:
    WindowProfiler* profiler_;
    int exceptions_;
  };

  ~WindowProfiler() noexcept {
    if (!finished_) {
      try {
        finish();
      } catch (const std::exception& error) {
        std::cerr << "PTExpt profiler finalization failed: " << error.what()
                  << std::endl;
      }
    }
  }

 private:
  WindowProfiler(std::string file, int skip, int active, bool gpu,
                 std::function<void()> synchronize)
      : file_(std::move(file)), skip_(skip), active_(active), gpu_(gpu),
        synchronize_(std::move(synchronize)) {}

  static int read_count(const char* name, int fallback, bool allow_zero) {
    const char* value = std::getenv(name);
    if (!value) {
      return fallback;
    }
    errno = 0;
    char* end = nullptr;
    long count = std::strtol(value, &end, 10);
    if (errno || end == value || *end || count < (allow_zero ? 0 : 1) ||
        count > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(name) +
                                  " must be a valid call count");
    }
    return static_cast<int>(count);
  }

  void begin() {
    if (depth_++ != 0) {
      return;
    }
    try {
      if (seen_ < skip_) {
        return;
      }
      if (!running_) {
        if (torch::autograd::profiler::profilerEnabled()) {
          throw std::runtime_error("PTExpt DP_PROFILER cannot overlap another profiler on the inference thread");
        }
        // Drain warmup work before recording any events from the active window.
        synchronize_();
        using namespace torch::profiler::impl;
        std::set<ActivityType> activities{ActivityType::CPU};
        if (gpu_) {
          activities.insert(ActivityType::CUDA);
        }
        ProfilerConfig config(ProfilerState::KINETO, false, false, false, false,
                              false, ExperimentalConfig());
        torch::autograd::profiler::prepareProfiler(config, activities);
        torch::autograd::profiler::enableProfiler(config, activities);
        running_ = true;
      }
      range_ = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
      range_->before("deepmd/pt_expt.compute");
    } catch (...) {
      // A Scope whose constructor fails has no destructor. Close any profiler
      // this instance started and make the failed window terminal before retry.
      depth_ = 0;
      aborted_ = true;
      try {
        finish();
      } catch (const std::exception& error) {
        std::cerr << "PTExpt profiler startup cleanup failed: " << error.what()
                  << std::endl;
      }
      throw;
    }
  }

  void end(bool failed) {
    aborted_ = aborted_ || failed;
    if (--depth_ != 0) {
      return;
    }
    range_.reset();
    ++seen_;
    if (running_ && !failed) {
      ++captured_;
    }
    if (aborted_ || captured_ == active_) {
      finish();
    }
  }

  void finish() {
    finished_ = true;
    range_.reset();
    if (running_) {
      // Always disable TLS profiler even when device synchronization fails.
      std::exception_ptr sync_error;
      try {
        synchronize_();
      } catch (...) {
        aborted_ = true;
        sync_error = std::current_exception();
      }
      running_ = false;
      auto result = torch::autograd::profiler::disableProfiler();
      if (result) {
        result->save(file_);
      }
      write_summary();
      if (sync_error) {
        std::rethrow_exception(sync_error);
      }
    } else {
      write_summary();
    }
  }

  void write_summary() const {
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(file_ + ".summary.json");
    output << "{\n  \"skip_calls\": " << skip_
           << ",\n  \"active_calls\": " << active_
           << ",\n  \"observed_calls\": " << seen_
           << ",\n  \"captured_calls\": " << captured_
           << ",\n  \"aborted\": " << (aborted_ ? "true" : "false")
           << ",\n  \"complete\": "
           << ((!aborted_ && captured_ == active_) ? "true" : "false")
           << "\n}\n";
    output.close();
  }

  std::string file_;
  int skip_;
  int active_;
  bool gpu_;
  std::function<void()> synchronize_;
  int seen_ = 0;
  int captured_ = 0;
  int depth_ = 0;
  bool running_ = false;
  bool finished_ = false;
  bool aborted_ = false;
  std::unique_ptr<at::RecordFunction> range_;
};

}  // namespace deepmd::ptexpt
