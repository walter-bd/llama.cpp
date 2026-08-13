# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# examples

if (EMSCRIPTEN)
else()
    add_subdirectory(batched)
    add_subdirectory(debug)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)
    add_subdirectory(attnres-sidecar)
add_subdirectory(attnres-address-space)
add_subdirectory(attnres-block-address-space)
add_subdirectory(injection-depth-threshold)
add_subdirectory(multi-injection-threshold)
add_subdirectory(injection-sensitivity-allocation)
add_subdirectory(injection-allocation-heldout)
add_subdirectory(teacher-modulated-plasticity)
add_subdirectory(teacher-curriculum-plasticity)
add_subdirectory(teacher-update-sanity)
add_subdirectory(teacher-retrieval-authority)
add_subdirectory(autonomous-plasticity-controller)
add_subdirectory(relation-geometry-probe)
add_subdirectory(forward-predictive-trace)
add_subdirectory(autonomous-sidecar-integration)
add_subdirectory(recurrence-write-consolidation)
add_subdirectory(forward-plasticity)
    add_subdirectory(forward-target-plasticity)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(idle)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)
    add_subdirectory(diffusion)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
