module cuda_profile
    use, intrinsic :: iso_c_binding
    implicit none

    ! CUDA interfaces
    ! The IMPORT statement makes named entities from the
    ! host scoping unit accessible in the interface body
    interface
        subroutine launch_cuda_loop(data, n) bind(C)
            import :: c_double, c_int
            real(c_double), dimension(*), intent(inout) :: data
            integer(c_int), value :: n
        end subroutine launch_cuda_loop

        subroutine launch_cuda_dgemm(m, n, k, A, B, C) bind(C)
            import :: c_double, c_int
            integer(c_int), value :: m, n, k
            real(c_double), dimension(*), intent(in) :: A, B
            real(c_double), dimension(*), intent(out) :: C
        end subroutine launch_cuda_dgemm
    end interface

    ! BLAS interfaces
    interface
        subroutine dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import :: c_double
            character, intent(in) :: transa, transb
            integer, intent(in) :: m, n, k, lda, ldb, ldc
            real(c_double), intent(in) :: alpha, beta
            real(c_double), intent(in) :: a(lda, *)
            real(c_double), intent(in) :: b(ldb, *)
            real(c_double), intent(inout) :: c(ldc, *)
        end subroutine dgemm
    end interface

    private
    
    real(c_double), parameter, public :: accuracy = 1.0e-6

    public :: profile_loop, profile_matrix

    contains

        !> @brief Adds 1.0 to each element of the input array in a loop on the CPU.
        !>
        !> This subroutine performs an element-wise addition of 1.0 to the input array `data`.
        !> The loop is executed on the CPU.
        !>
        !> @param[inout] data: A one-dimensional array of real numbers (double precision) to be modified.
        !> @param[in] n: The size of the input data array.
        subroutine cpu_loop(data, n)
           real(c_double), dimension(:), intent(inout) :: data
           integer, intent(in) :: n
           integer :: i

           do i = 1, n
               data(i) = data(i) + 1.0
           end do
       end subroutine cpu_loop

       !> @brief Profiles and compares the execution times of a loop on CPU and GPU.
       !>
       !> This subroutine runs a loop on the CPU using the `cpu_loop` subroutine and on the
       !> GPU using the `launch_cuda_loop` subroutine. It measures the execution times for
       !> both loops and compares the accuracy of the results. 
       !>
       !> @param[in] n_loop: The number of loop iterations or the size of the data arrays.
       subroutine profile_loop(n_loop)
           integer, intent(in)               :: n_loop

           ! Local variables
           integer                           :: i
           real(c_double), dimension(n_loop) :: data_cpu, data_gpu
           real(c_double)                    :: start_time, elapsed_time_cpu,&
                                                elapsed_time_gpu

           ! Initialize data
           data_cpu = 0.0
           data_gpu = 0.0

           ! Launch the CPU loop
           call cpu_time(start_time)
           call cpu_loop(data_cpu, n_loop)
           call cpu_time(elapsed_time_cpu)
           elapsed_time_cpu = elapsed_time_cpu - start_time

           ! Launch the GPU loop
           call cpu_time(start_time)
           call launch_cuda_loop(data_gpu, n_loop)
           call cpu_time(elapsed_time_gpu)
           elapsed_time_gpu = elapsed_time_gpu - start_time

           ! Compare the results and print the execution times
           do concurrent (i = 1:n_loop)
               if (abs(data_cpu(i) - data_gpu(i)) > accuracy) then
                   print *, "Vector mismatch at index ", i, ":", data_cpu(i), "vs", data_gpu(i)
               end if
           end do
           print *, "Loop execution time (CPU):", elapsed_time_cpu, "seconds"
           print *, "Loop execution time (GPU):", elapsed_time_gpu, "seconds"
       end subroutine profile_loop

       !> @brief Profiles and compares the execution times of matrix multiplication on CPU and GPU.
       !>
       !> This subroutine performs matrix multiplication using the BLAS `dgemm` subroutine on the
       !> CPU and the cuBLAS wrapper `launch_cuda_dgemm` on the GPU. It measures the execution
       !> times for both operations and compares the accuracy of the results.
       !> C := alpha * A * B + beta * C
       !>
       !> @param[in] m_mat: Number of rows in matrix A and matrix C.
       !> @param[in] n_mat: Number of columns in matrix B and matrix C.
       !> @param[in] k_mat: Number of columns in matrix A and number of rows in matrix B.
       subroutine profile_matrix(m_mat, n_mat, k_mat)
           integer, intent(in) :: m_mat, n_mat, k_mat

           ! Internal variables
           integer                                   :: i, j
           real(c_double)                            :: start_time, elapsed_time_cpu,&
                                                        elapsed_time_gpu
           ! For cuBLAS
           real(c_double), dimension(m_mat * k_mat)  :: A_GPU
           real(c_double), dimension(k_mat * n_mat)  :: B_GPU
           real(c_double), dimension(m_mat * n_mat)  :: C_GPU
           real(c_double)                            :: alpha, beta

           ! For BLAS
           real(c_double), dimension(m_mat, k_mat)   :: A_CPU
           real(c_double), dimension(k_mat, n_mat)   :: B_CPU
           real(c_double), dimension(m_mat, n_mat)   :: C_CPU

           ! Initialize matrices
           A_GPU = [(i * 1.0, i = 1, m_mat * k_mat)]
           B_GPU = [(i * 2.0, i = 1, k_mat * n_mat)]
           A_CPU = reshape(A_GPU, [m_mat, k_mat])
           B_CPU = reshape(B_GPU, [k_mat, n_mat])
           C_CPU = 0.0
           C_GPU = 0.0

           ! Run BLAS
           alpha = 1.0
           beta = 0.0
           call cpu_time(start_time)
           call dgemm('N', 'N', m_mat, n_mat, k_mat, alpha, A_CPU, m_mat, B_CPU, k_mat, beta, C_CPU, m_mat)
           call cpu_time(elapsed_time_cpu)
           elapsed_time_cpu = elapsed_time_cpu - start_time

           ! Run cuBLAS
           call cpu_time(start_time)
           call launch_cuda_dgemm(m_mat, n_mat, k_mat, A_GPU, B_GPU, C_GPU)
           call cpu_time(elapsed_time_gpu)
           elapsed_time_gpu = elapsed_time_gpu - start_time

           ! Compare the results and print the execution times
           do j = 1, n_mat
              do i = 1, m_mat
                 if (abs(C_CPU(i, j) - C_GPU(i + (j - 1) * m_mat)) > accuracy) then
                    print *, "Matrix mismatch at index (", i, ",", j, "):", C_CPU(i, j), "vs", C_GPU(i + (j - 1) * m_mat)
                 end if
              end do
           end do
           print *, "Matrix multiplication execution time (CPU):", elapsed_time_cpu, "seconds"
           print *, "Matrix multiplication execution time (GPU):", elapsed_time_gpu, "seconds"

       end subroutine profile_matrix

end module cuda_profile

program main
    use cuda_profile, only: profile_loop, profile_matrix
    implicit none
    integer :: n_loop, m_mat, n_mat, k_mat

    ! Matrix and loop sizes
    m_mat = 150
    n_mat = 1000
    k_mat = 200
    n_loop = 1e8

    ! Loop profiling
    call profile_loop(n_loop)

    ! Matrix multiplication profiling
    call profile_matrix(m_mat, n_mat, k_mat)

end program main
