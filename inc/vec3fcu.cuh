//
// Created by lidan on 4/11/2020.
//

#ifndef OBJVIEWER_VEC3CU_CUH
#define OBJVIEWER_VEC3CU_CUH


#include <ostream>
#include <math.h>
#include <cuda_runtime.h>


#include "helper_cuda.h"

#include "vec3f.h"
#include "forceline.h"
#define     GLH_ZERO                double(0.0)
#define     GLH_EPSILON          double(10e-6)
#define		GLH_EPSILON_2		double(10e-12)
#define     equivalent(a,b)             (((a < b + GLH_EPSILON) &&\
                                                      (a > b - GLH_EPSILON)) ? true : false)
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


inline __device__ __host__ double fmaxmy(double a, double b, double c) {
//    return (a > b) ? a : b;
    return fmax(fmax(a,b),c) ;
}

inline __device__ __host__ double fminmy(double a, double b, double c) {
//    return (a < b) ? a : b;
    return fmin(fmin(a,b),c) ;
}

//inline __device__  bool isEqual( double a, double b, double tol=GLH_EPSILON )
//{
//    return fabs( a - b ) < tol;
//}

/* This is approximately the smallest number that can be
* represented by a float, given its precision. */
#define ALMOST_ZERO		FLT_EPSILON

#ifndef M_PI
#define M_PI 3.14159f
#endif

#include <assert.h>
inline __host__ __device__ double dmax(double a, double b) {
    return (a > b) ? a : b;
}

inline  __host__ __device__ double dmin(double a, double b) {
    return (a < b) ? a : b;
}



struct vec2fcu {
    union {
        struct {
            double x, y;
        };
        struct {
            double v[2];
        };
    };
public:

    FORCEINLINE __host__ __device__ vec2fcu ()
    {x=0; y=0;}

    FORCEINLINE __host__ __device__ vec2fcu(const vec2fcu &v)
    {
        x = v.x;
        y = v.y;
    }

    FORCEINLINE __host__ __device__ vec2fcu(const double *v)
    {
        x = v[0];
        y = v[1];
    }

    FORCEINLINE __host__ __device__ vec2fcu(double x, double y)
    {
        this->x = x;
        this->y = y;
    }

    FORCEINLINE __host__ __device__ vec2fcu  &operator=(vec2fcu other) {
        x = other.x ;
        y = other.y ;
    }

    FORCEINLINE __host__ __device__ double operator [] ( int i ) const {return v[i];}
    FORCEINLINE __host__ __device__ double &operator [] (int i) { return v[i]; }


};


struct vec3icu{
    union {
        struct {
            unsigned int x, y, z;
        };
        struct {
            unsigned int v[3];
        };
    };

    FORCEINLINE  __host__ __device__ vec3icu(unsigned int x, unsigned int y, unsigned int z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }


    FORCEINLINE __host__ __device__ vec3icu ()
    {x=0; y=0; z=0;}

    FORCEINLINE __host__ __device__ vec3icu(const vec3icu &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    FORCEINLINE __host__ __device__  unsigned int operator [] ( int i ) const {return v[i];}
    FORCEINLINE __host__ __device__ unsigned int &operator [] (int i) { return v[i]; }

    FORCEINLINE __host__ __device__ vec3icu &operator += (const vec3icu &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    FORCEINLINE __host__ __device__ vec3icu &operator =(vec3icu other)
    {
        x = other.x ;
        y = other.y ;
        z = other.z ;
        return *this ;
    }


};
struct vec3fcu {\

public:
    union {
        struct {
            double x, y, z;
        };
        struct {
            double v[3];
        };


    };

    FORCEINLINE __host__ __device__ vec3fcu ()
    {x=0; y=0; z=0;}

    FORCEINLINE __host__ __device__ vec3fcu(const vec3fcu &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    FORCEINLINE __host__ __device__ vec3fcu(const vec3f &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }


    FORCEINLINE __host__ __device__ vec3fcu(const double *v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    FORCEINLINE  __host__ __device__ vec3fcu(double x, double y, double z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    FORCEINLINE __host__ __device__  double operator [] ( int i ) const {return v[i];}
    FORCEINLINE __host__ __device__ double  operator [] (int i) { return v[i]; }

    FORCEINLINE __host__ __device__ vec3fcu &operator += (const vec3fcu &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    FORCEINLINE __host__ __device__ vec3fcu &operator -= (const vec3fcu &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    FORCEINLINE __host__ __device__ vec3fcu &operator *= (double t) {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    FORCEINLINE __host__ __device__ vec3fcu &operator /= (double t) {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    FORCEINLINE __host__ __device__ vec3fcu &operator=(vec3f other)
    {
        x = other.x ;
        y = other.y ;
        z = other.z ;
        return *this ;
    }

    FORCEINLINE __host__ __device__ vec3fcu &operator=(vec3fcu other)
    {
        x = other.x ;
        y = other.y ;
        z = other.z ;
        return *this ;
    }
    FORCEINLINE __host__ __device__ bool operator == (const vec3fcu& other)const
    {
        return x==other.x && y==other.y && z==other.z ;
    }


    FORCEINLINE __host__ __device__ void negate() {
        x = -x;
        y = -y;
        z = -z;
    }

    FORCEINLINE __host__ __device__ vec3fcu operator - () const {
        return vec3fcu(-x, -y, -z);
    }

    FORCEINLINE __host__ __device__ bool operator> (const vec3fcu& cu) const {
        return x>cu.x&&y>cu.y&&z>cu.z ;
    }
    FORCEINLINE __host__ __device__ bool operator< (const vec3fcu& cu) const {
        return x<cu.x&&y<cu.y&&z<cu.z ;
    }


    FORCEINLINE __host__ __device__ vec3fcu operator+ (const vec3fcu &v) const
    {
        return vec3fcu(x+v.x, y+v.y, z+v.z);
    }

    FORCEINLINE __host__ __device__ vec3fcu operator- (const vec3fcu &v) const
    {
        return vec3fcu(x-v.x, y-v.y, z-v.z);
    }

    FORCEINLINE __host__ __device__ vec3fcu operator *(double t) const
    {
        return vec3fcu(x*t, y*t, z*t);
    }

    FORCEINLINE __host__ __device__ vec3fcu operator /(double t) const
    {
        return vec3fcu(x/t, y/t, z/t);
    }

    FORCEINLINE __host__ __device__ vec3fcu vmax(vec3fcu another)  const
    {
        return vec3fcu(dmax(x,another.x), dmax(y,another.y), dmax(z,another.z));
    }

    FORCEINLINE __host__ __device__ vec3fcu vmin(vec3fcu another)  const
    {
        return vec3fcu(dmin(x,another.x), dmin(y,another.y), dmin(z,another.z));
    }



    // cross product
    FORCEINLINE __host__ __device__ const vec3fcu cross(const vec3fcu &vec) const
    {
        return vec3fcu(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
    }

    FORCEINLINE __host__ __device__ double dot(const vec3fcu &vec) const {
        return x*vec.x+y*vec.y+z*vec.z;
    }

    FORCEINLINE __host__ __device__ void normalize()
    {
        double sum = x*x+y*y+z*z;
        if (sum > GLH_EPSILON_2) {
            double base = double(1.0/sqrt(sum));
            x *= base;
            y *= base;
            z *= base;
        }
    }

    FORCEINLINE __host__ __device__ double length() const {
        return double(sqrt(x*x + y*y + z*z));
    }

    FORCEINLINE __host__ __device__ vec3fcu getUnit() const {
        return (*this)/length();
    }

    FORCEINLINE __host__ __device__ bool isUnit() const {
        return isEqual( squareLength(), 1.f );
    }

    //! max(|x|,|y|,|z|)
    FORCEINLINE __host__ __device__ double infinityNorm() const
    {
        return fmax(fmax( fabs(x), fabs(y) ), fabs(z));
    }

    FORCEINLINE __host__ __device__ vec3fcu & set_value( const double &vx, const double &vy, const double &vz)
    { x = vx; y = vy; z = vz; return *this; }


    FORCEINLINE __host__ __device__ bool equal_abs(const vec3fcu &other) {
        return x == other.x && y == other.y && z == other.z;
    }

    FORCEINLINE __host__ __device__ double squareLength() const {
        return x*x+y*y+z*z;
    }

    static __host__ __device__ vec3fcu zero() {
        return vec3fcu(0.f, 0.f, 0.f);
    }

    //! Named constructor: retrieve vector for nth axis
    static __host__ __device__ vec3fcu axis( int n ) {
        assert( n < 3 );
        switch( n ) {
            case 0: {
                return xAxis();
            }
            case 1: {
                return yAxis();
            }
            case 2: {
                return zAxis();
            }
        }
        return vec3fcu();
    }
    void  __host__ __device__ abs()
    {
        x = x>=0?x:-x ;
        y = y>=0?y:-y ;
        z = z>=0?z:-z ;
    }
    //! Named constructor: retrieve vector for x axis
    static __host__ __device__ vec3fcu xAxis() { return vec3fcu(1.f, 0.f, 0.f); }
    //! Named constructor: retrieve vector for y axis
    static __host__ __device__ vec3fcu yAxis() { return vec3fcu(0.f, 1.f, 0.f); }
    //! Named constructor: retrieve vector for z axis
    static __host__ __device__ vec3fcu zAxis() { return vec3fcu(0.f, 0.f, 1.f); }

};

inline __host__ __device__ vec3fcu operator * (double t, const vec3fcu &v) {
    return vec3fcu(v.x*t, v.y*t, v.z*t);
}

inline __host__ __device__ vec3fcu interp(const vec3fcu &a, const vec3fcu &b, double t)
{
    return a*(1-t)+b*t;
}

inline __host__ __device__ vec3fcu vinterp(const vec3fcu &a, const vec3fcu &b, double t)
{
    return a*t+b*(1-t);
}

inline __host__ __device__ vec3fcu interp(const vec3fcu &a, const vec3fcu &b, const vec3fcu &c, double u, double v, double w)
{
    return a*u+b*v+c*w;
}

//inline __host__ __device__ double clamp(double f, double a, double b)
//{
//    return fmax(a, fmin(f, b));
//}

inline __host__ __device__ double vdistance(const vec3fcu &a, const vec3fcu &b)
{
    return (a-b).length();
}

inline __host__ __device__ std::ostream& operator<<( std::ostream&os, const vec3fcu &v ) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    return os;
}

#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)




#endif //OBJVIEWER_VEC3CU_CUH
