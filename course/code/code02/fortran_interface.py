%%writefile fortran_interface.f90
module fortran_module
    use iso_c_binding, only: c_double
    implicit none
contains
    function f_sin_cos(x) result(f) bind(C, name="f_sin_cos")
        implicit none
        real(c_double), intent(in) :: x
        real(c_double) :: f
        f = sin(x) * cos(x)
    end function f_sin_cos
end module fortran_module