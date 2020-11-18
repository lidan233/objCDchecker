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

#include "vec3f.h"
#include "mat3f.h"

#include "tri.h"
#include "edge.h"
#include "box.h"
#include <set>
using namespace std;

class mesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	
	tri3f *_tris;

	unsigned int _num_edge;
	edge4f *_edges;
	double *_rest_lengths;

	// used by time integration
	vec3f *_vtxs;
	vec3f *_ivtxs; // initial positions
	vec3f *_ovtxs; // previous positions

	vec3f *_nrms;
	vec2f *_texs;
	bool _first;

	mesh(unsigned int numVtx, unsigned int numTri, unsigned int numEdge, tri3f *tris, edge4f *edges, vec3f *vtxs, vec2f *texs=NULL);
	~mesh();

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f *getOVtxs() const { return _ovtxs;}
	vec3f *getIVtxs() const {return _ivtxs;}

	void update(matrix3f &, vec3f &);

	// calc norms, and prepare for display ...
	void updateNrms();

	void getRestLength();

	// really displaying ...
	void display(bool tri, bool pnt, bool edge, int level, bool rigid, set<int>&, int);

	// povray file output
	void povray(char *fname, bool first);

	// obj file putput
	void exportObj(char *fname, bool cloth, int id);

	// load vtxs
	bool load(FILE *fp);

	BOX bound() {
		BOX a;

		for (int i=0; i<_num_vtx; i++)
			a += _vtxs[i];

		return a;
	}
};
