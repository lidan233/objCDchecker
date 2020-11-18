//**************************************************************************************
//  Copyright (C) 2017 - 2022, Min Tang
//  All rights reserved.
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//     1. Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//     2. Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in the
//        documentation and/or other materials provided with the distribution.
//     3. The names of its contributors may not be used to endorse or promote
//        products derived from this software without specific prior written
//        permission.
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//**************************************************************************************

#pragma once

#include <ostream>
#include <math.h>
#include <string>
#include <fstream>
#include <set>
#include <vector>
#include <cuda_runtime.h>

#include "forceline.h"

#define     GLH_ZERO                double(0.0)
#define     GLH_EPSILON          double(10e-6)
#define		GLH_EPSILON_2		double(10e-12)
#define     equivalent(a,b)             (((a < b + GLH_EPSILON) &&\
                                                      (a > b - GLH_EPSILON)) ? true : false)

inline __host__ __device__ double fmax(double a, double b) {
	return (a > b) ? a : b;
}

inline double __host__ __device__ fmin(double a, double b) {
	return (a < b) ? a : b;
}

inline bool isEqual( double a, double b, double tol=GLH_EPSILON )
{
    return fabs( a - b ) < tol;
}

/* This is approximately the smallest number that can be
* represented by a float, given its precision. */
#define ALMOST_ZERO		FLT_EPSILON

#ifndef M_PI
#define M_PI 3.14159f
#endif

#include <assert.h>
#include <algorithm>

class vec2f {
	union {
		struct {
		double x, y;
		};
		struct {
		double v[2];
		};
	};
public:

	FORCEINLINE vec2f ()
	{x=0; y=0;}

	FORCEINLINE vec2f(const vec2f &v)
	{
		x = v.x;
		y = v.y;
	}

	FORCEINLINE vec2f(const double *v)
	{
		x = v[0];
		y = v[1];
	}

	FORCEINLINE vec2f(double x, double y)
	{
		this->x = x;
		this->y = y;
	}

	FORCEINLINE double operator [] ( int i ) const {return v[i];}
	FORCEINLINE double &operator [] (int i) { return v[i]; }
};

class vec3f {
public:
	union {
		struct {
		double x, y, z;
		};
		struct {
		double v[3];
		};
	};


	FORCEINLINE __host__ __device__ vec3f ()
	{x=0; y=0; z=0;}

	FORCEINLINE __host__ __device__ vec3f(const vec3f &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	FORCEINLINE __host__ __device__ vec3f(const double *v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	FORCEINLINE __host__ __device__ vec3f(double x, double y, double z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	FORCEINLINE __host__ __device__ double operator [] ( int i ) const {return v[i];}
	FORCEINLINE __host__ __device__ double &operator [] (int i) { return v[i]; }
    FORCEINLINE __host__ __device__ bool operator == (const vec3f &v) {
        if(x==v.x && y==v.y && z==v.z)
            return true;
        return false;
    }

	FORCEINLINE __host__ __device__ vec3f &operator += (const vec3f &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	FORCEINLINE __host__ __device__ vec3f &operator -= (const vec3f &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	FORCEINLINE __host__ __device__ vec3f &operator *= (double t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	FORCEINLINE __host__ __device__ vec3f &operator /= (double t) {
		x /= t;
		y /= t;
		z /= t;
		return *this;
	}

	FORCEINLINE __host__ __device__ void negate() {
		x = -x;
		y = -y;
		z = -z;
	}

	FORCEINLINE __host__ __device__ vec3f operator - () const {
		return vec3f(-x, -y, -z);
	}

	FORCEINLINE __host__ __device__ vec3f operator+ (const vec3f &v) const
	{
		return vec3f(x+v.x, y+v.y, z+v.z);
	}

	FORCEINLINE __host__ __device__ vec3f operator- (const vec3f &v) const
	{
		return vec3f(x-v.x, y-v.y, z-v.z);
	}

	FORCEINLINE __host__ __device__ vec3f operator *(double t) const
	{
		return vec3f(x*t, y*t, z*t);
	}

	FORCEINLINE __host__ __device__ vec3f operator /(double t) const
	{
		return vec3f(x/t, y/t, z/t);
	}

     // cross product
     FORCEINLINE __host__ __device__ const vec3f cross(const vec3f &vec) const
     {
          return vec3f(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
     }

	 FORCEINLINE __host__ __device__ double dot(const vec3f &vec) const {
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

	 FORCEINLINE __host__ __device__ vec3f getUnit() const {
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

	FORCEINLINE __host__ __device__ vec3f & set_value( const double &vx, const double &vy, const double &vz)
	{ x = vx; y = vy; z = vz; return *this; }

	FORCEINLINE __host__ __device__ bool equal_abs(const vec3f &other) {
		return x == other.x && y == other.y && z == other.z;
	}

	FORCEINLINE __host__ __device__ double squareLength() const {
		return x*x+y*y+z*z;
	}

	static __host__ __device__ vec3f zero() {
		return vec3f(0.f, 0.f, 0.f);
	}

    //! Named constructor: retrieve vector for nth axis
	static __host__ __device__ vec3f axis( int n ) {
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
		return vec3f();
	}

    //! Named constructor: retrieve vector for x axis
	static __host__ __device__ vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
    //! Named constructor: retrieve vector for y axis
	static __host__ __device__ vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
    //! Named constructor: retrieve vector for z axis
	static __host__ __device__ vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

};

inline __host__ __device__ vec3f operator * (double t, const vec3f &v) {
	return vec3f(v.x*t, v.y*t, v.z*t);
}

inline __host__ __device__ vec3f interp(const vec3f &a, const vec3f &b, double t)
{
	return a*(1-t)+b*t;
}

inline __host__ __device__ vec3f vinterp(const vec3f &a, const vec3f &b, double t)
{
	return a*t+b*(1-t);
}

inline __host__ __device__ vec3f interp(const vec3f &a, const vec3f &b, const vec3f &c, double u, double v, double w)
{
	return a*u+b*v+c*w;
}

inline __host__ __device__ double clamp(double f, double a, double b)
{
	return fmax(a, fmin(f, b));
}

inline __host__ __device__ double vdistance(const vec3f &a, const vec3f &b)
{
	return (a-b).length();
}

inline __host__ __device__ std::ostream& operator<<( std::ostream&os, const vec3f &v ) {
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	return os;
}

template<typename type>
inline void set_to_file(std::set<type> data,std::string file)
{
    std::ofstream out(file.c_str()) ;
    std::vector<type> t(data.size()) ;
    std::copy(data.begin(),data.end(),t.begin()) ;


    std::sort(t.begin(),t.end(),[](const type& a,const type& b){
        return a>b;
    }) ;
    if (out.is_open())
    {
        out<<t.size()<<std::endl ;

        for(auto i = t.begin() ; i != t.end();i++)
        {
            out<< (*i) <<std::endl ;
        }
        out.close() ;
    }

}


#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)
