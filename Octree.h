

#define _M_SSE

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <typeinfo>
#pragma once

	static const char lowhighMASK00000_[]={0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7};
	static const char highlowMASK00000_[]={8,9,10,11,12,13,14,15,8,9,10,11,12,13,14,15};
#if defined(_M_NEON)

	#define f4v float32x4_t
	#define u4v uint32x4_t
	#define f4_and(x,y) vandq_s32((u4v)x,(u4v)y)

	#define f4_xor(x,y) veorq_s32((u4v)x,(u4v)y)
	#define f4_add(x,y) vaddq_f32(x,y)
	#define f4_sub(x,y) vsubq_f32(x,y)
	#define f4_mul(x,y) vmulq_f32(x,y)
	#define f4_le(x,y) vcleq_f32(x,y)
	#define f4_lt(x,y) vcltq_f32(x,y)
	#define f4_ge(x,y) vcgeq_f32(x,y)
	#define f4_gt(x,y) vcgtq_f32(x,y)
	#define f4_eq(x,y) vceqq_f32(x,y)
	#define f4_div(x,y) vdivq_f32(x,y)
	#define f4_ld(x) vld1q_f32(x)
	#define f4_st(x) vst1q_f32(x)
	#define f4_max(x,y) vmaxq_f32(x,y)
	#define f4_min(x,y) vminq_f32(x,y)
	#define u4_hadd(x,y) vaddq_s32(x,y)
	#define u4_add(x,y) vaddq_s32(x,y)
	#define u4_sub(x,y) vsubq_s32(x,y)
	#define u4_mul(x,y) vmulq_s32(x,y)
	#define u4_ld(x) vld1q_s32(x)
	#define u4_or(x,y) vorq_u32(x,y)
	#define u4_st(x,y) vst1q_s32(x,y)
	#define u4_pshufb(x,y) vtblq_s32(x,y)
	#define f4_unpackhi(x,y) vcombine_f32(vget_high_f32(x),vget_high_f32(y))
	#define f4_unpacklo(x,y) vcombine_f32(vget_low_f32(x),vget_low_f32(y))
	#define f4_movehl(x,y) vtblq_s32((u4v)x,vld1q_s32(highlowMASK00000_))
	#define f4_movelh(x,y) vtblq_s32((u4v)x,vld1q_s32(lowhighMASK00000_))

#elif defined(_M_SSE)

	#define f4v __m128
	#define u4v __m128i
	#define f4_add(x,y) _mm_add_ps(x,y)
	#define f4_sub(x,y) _mm_sub_ps(x,y)
	#define f4_mul(x,y) _mm_mul_ps(x,y)
	#define f4_le(x,y) _mm_cmple_ps(x,y)
	#define f4_lt(x,y) _mm_cmplt_ps(x,y)
	#define f4_ge(x,y) _mm_cmpge_ps(x,y)
	#define f4_gt(x,y) _mm_cmpgt_ps(x,y)
	#define f4_eq(x,y) _mm_cmpeq_ps(x,y)
	#define f4_div(x,y) _mm_div_ps(x,y)
	#define f4_ld(x) _mm_loadu_ps(x)
	#define f4_st(x,y) _mm_storeu_ps(x,y)
	#define f4_and(x,y) _mm_and_ps(x,y)
	#define f4_max(x,y) _mm_max_ps(x,y)
	#define f4_min(x,y) _mm_min_ps(x,y)
	#define u4_hadd(x,y) _mm_hadd_epi32(x,y)
	#define u4_add(x,y) _mm_add_epi32(x,y)
	#define u4_sub(x,y) _mm_sub_epi32(x,y)
	#define u4_mul(x,y) _mm_cvtps_epi32((f4v)_mm_mul_ps((f4v)_mm_cvtepi32_ps((__m128i)x),(f4v)_mm_cvtepi32_ps((__m128i)y)))
	#define u4_xor(x,y) (u4v)_mm_xor_ps((f4v)x,(f4v)y)
	#define u4_ld(x) _mm_loadu_epi32(x)
	#define u4_st(x,y) _mm_storeu_ps(x,y)
	#define u4_or(x,y) _mm_or_ps((f4v)x, (f4v)y)
	#define u4_pshufb(x,y) _mm_shuffle_epi8(x,y)
#define f4_unpackhi(x,y) _mm_unpackhi_ps(x,y)
	#define f4_unpacklo(x,y) _mm_unpacklo_ps(x,y)
	#define f4_movehl(x,y) _mm_movehl_ps(x,y)
	#define f4_movelh(x,y) _mm_movelh_ps(x,y)


#endif
typedef int ivector_init __attribute__((vector_size(16)));
template<unsigned char levels=12, unsigned int max_items=1, int siz=32*1024*1024> class Octree{
public:
	struct header{
		int size_x;
		int size_y;
	}; //we want to be able to treat the octree as a giant bounding volume, without polluting the cache with all this other stuff.
	//and also without aliasing.

	struct node{
		unsigned int n;
		short children_offset;
		unsigned char data[15];
		short parent_offset;
		unsigned char level;
		unsigned long N; //the octree is its own pool allocator.
	};//remember to align this so each node is half a cache line.
	header space_size;
	size_t N, memsz;
	node*    mTreeRoot;
	__attribute__((aligned(32))) node pool[siz/sizeof(node)];
	Octree(){
		memsz=siz;
		mTreeRoot=(node*)pool;
	}
	~Octree(){}
	void clear(){
		N=1;
		memset(pool, 0, memsz);
		mTreeRoot->N=0;
		mTreeRoot->n=0;
	}
	inline node* traverseandget(float coords[4]){
		node*   nd=mTreeRoot;

		f4v c=f4_ld(coords);
		f4v center={0.5,0.5,0.5,0.5};

		for(unsigned char lvl=0; lvl<levels; ++lvl){
			if(!(nd->children_offset)){
							break;
			}
			unsigned ix;

#if !defined(_M_SSE) || !defined(_LP64)
					const u4v indvec = __extension__ (u4v)(ivector_init){0,4,2,1};
					const u4v mask = __extension__ (u4v)(ivector_init){0,1,1,1};
		   			const u4v unmask =__extension__(u4v)(ivector_init){0,0,0,0};
		   			f4v halfmask=__extension__ (f4v){0.5,0.5,0.5,0.5};
		   			f4v fmask=__extension__(f4v){1.5,1.5,1.5,1.5};
					f4v c=f4_ld(coords);
					f4v center ={0.5,0.5,0.5,0.5};
					u4v test=(u4v)f4_and((f4v)f4_le (c, center), (f4v)mask);
					test=u4_mul((u4v)test, (u4v)indvec);
					test=u4_hadd(test, test);
					test=u4_hadd(test, test);
					unsigned int indices[4];
					f4_st((float*)indices, (f4v)test);
					ix=indices[0];
			    		  int off=(nd->children_offset+ix);
			    		   			nd=&nd[off];

			    		   					u4v gt =(u4v)f4_and((f4v)f4_le((f4v)center, (f4v)c), (f4v)mask);
			    		   					u4v lt = u4_xor((u4v)gt,(u4v) mask);
			    		   					f4v flt=(f4v)u4_mul((u4v)lt, (u4v)halfmask);

			    		   					f4v fgt=(f4v)u4_mul((u4v)lt,(u4v)fmask);


			    		   					center=f4_mul( f4_add(fgt,flt),center);
#else
			    		   __m128 indvec=__extension__ (__m128)(__v4si){ 0, 4, 2, 1 };
					asm volatile(
							"movaps %1, %%xmm7\n\t"
							"cmpleps %2, %%xmm7\n\t"
							"mov $0x1, %0\n\t"
							"movd %0, %%xmm14\n\t"
							"shufps $0x0, %%xmm14, %%xmm14\n\t"
							"andps %%xmm14, %%xmm7\n\t"
							"movaps %3, %%xmm2\n\t"
							"mulss %%xmm2, %%xmm7\n\t"
							"xorps %%xmm15, %%xmm15\n\t"
							"phaddd %%xmm7, %%xmm7\n\t"
							"phaddd %%xmm15, %%xmm7\n\t"
							"movd %%xmm7, %0\n\t"
							: "=r"(ix)
							:"x"(center),"x"(c),"x"(indvec)
							 :
						);

			int off=(nd->children_offset+ix);
			nd=&nd[off];
		asm volatile(
			"mov $0x1, %2\n\t"
			"cvtsi2ss %%ecx, %%xmm14\n\t"
			"movaps %%xmm14, %%xmm7\n\t"
			"addps %%xmm7, %%xmm7\n\t"
			"movaps %%xmm14, %%xmm8\n\t"
			"divps %%xmm7, %%xmm8\n\t"
			"movaps %0, %%xmm9\n\t"
			"cmpleps %1, %%xmm9\n\t"
			"andps %%xmm14, %%xmm9\n\t"
			"xorps %%xmm9, %%xmm14\n\t"
			"mov $0x3f800000, %2\n\t"
			"movd %2, %%xmm15\n\t"
			"shufps $0, %%xmm15, %%xmm15\n\t"
			"mulps %%xmm15, %%xmm9\n\t"
			"mulps %%xmm14, %%xmm9\n\t"
			"mulps %%xmm8, %%xmm14\n\t"
			"mulps %%xmm8, %%xmm7\n\t"
			"addps %%xmm8, %%xmm7\n\t"
			"mulps %%xmm7, %%xmm9\n\t"
			"addps %%xmm14, %%xmm9\n\t"
			"mulps %%xmm9, %0\n\t"
			:
			:"x"(center), "x"(c), "r"(off)
			:
		);
#endif
			/*__m128 zero=_mm_setzero_ps();
		__m128 one=_mm_set1_ps(1.);
		__m128 gt = _mm_andnot_ps(zero, _mm_cmpgt_ps(c, center));
		__m128 lt = _mm_andnot_ps(gt, one);
		__m128 centermul = _mm_add_ps(_mm_mul_ps(gt, mulgt),_mm_mul_ps(lt, mullt));
		center=_mm_mul_ps(centermul,center);*/

		}
		return nd;
	}
	char* filedata;
	int offset0;
	inline node* addsomething(float coords[4], char dat[15]){
				node *  ndp=traverseandget(coords);
				node nd=*ndp;
				unsigned int off;
					if(nd.n<max_items){
						for(int i=0; i<13; ++i){
							nd.data[i]+=dat[i];
						}
						nd.n++;
						*ndp=nd;
					}
					else{
							off=(N)*8;
							if(off<memsz){
							size_t newnd=(size_t)(&mTreeRoot[off]);
							int cond=((size_t)newnd<(size_t)mTreeRoot+memsz);
							off*=cond;
							for(int i=0; i<15; ++i){
								((node* )newnd)->data[i]+=dat[i];
							}
							((node* )newnd)->level=nd.level+1;
							((node* )newnd)->n=nd.n+1;
							ndp->n=0;
							register short parent_offset=((size_t)ndp-(size_t)newnd)/(sizeof(node));
							((node*  )ndp)->children_offset=-1*parent_offset;
							((node*)newnd)->parent_offset=parent_offset;
							(ndp)=(node*)newnd;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							((node* )(newnd+=sizeof(node)))->parent_offset=--parent_offset;
							//cond=(((node*  )newnd)->N);
							N=(N+1);
							}
					}
					return ndp;
	}
		inline void deletesomething(float coords[4], char data[15]){
			node*   ndp=traverseandget(coords);
			node nd=*ndp;
			--(nd.n);
			for(int b=0; b<15; ++b){
				if(data[b]>0) break;
				nd.data[b]-=data[b];
			}
			*ndp=nd;
			if(nd.n<=0){
				memset(ndp, 0, sizeof(node));
				unsigned int off=nd.parent_offset;
				node*   newnd=(ndp-off*sizeof(node));
				nd=*newnd;
				for(unsigned char ix=0; ix<8; ++ix){
					nd.n+=*((unsigned int*)(newnd+(nd.children_offset+ix)*sizeof(node)));
				}
				nd.children_offset=0;
				ndp->N=N;
				N=(ndp-mTreeRoot)/(8*sizeof(node));
			}
		}
		inline node* raycast(float o[], float d[], float* t){ //simple method, currently the parametric method needs a bit of work.
			node*   ndp;
			*t=0;
			while(!((ndp=(traverseandget(o)))->n)){
				float dst=(1<<ndp->level);
				o[0]+=d[0]/dst;
				o[1]+=d[1]/dst;
				o[2]+=d[2]/dst;
				*t+=1./dst;
				if(o[0]>1.) break;
				if(o[1]>1.) break;
				if(o[2]>1.) break;
			}
			return ndp;
		}
		inline int rayparam(float minf[4], float maxf[4], float szf[4],float df[4], float of[4], float af[4], unsigned char plane, float* t, node** hit){
			f4v min=f4_ld(minf);
			f4v max=f4_ld(maxf);
			f4v sz=f4_ld(szf);
			f4v d=f4_ld(df);
			f4v o=f4_ld(of);
			f4v a=f4_ld(af);
			unsigned cmp[4]={0,0,0,0};
			unsigned a0=0;
#if defined(_M_SSE) && defined(_LP64)
			volatile register __m128 mask=__extension__ (__m128)(__f4v){ 0.,4.,2.,1.};
			__asm volatile (
					"xorps %%xmm15,%%xmm15\n\t"
					"movaps %0, %%xmm14\n\t"
					"cmpltps %%xmm15, %%xmm14\n\t"
					"mov $0x1, %%eax\n\t"
					"movd %%eax, %%xmm15\n\t"
					"shufps $0, %%xmm15, %%xmm15\n\t"
					"andps %%xmm15, %%xmm14\n\t"
					"xorps %%xmm15,%%xmm15\n\t"
					"movaps %0, %%xmm13\n\t"
					"subps %%xmm13, %%xmm15\n\t"
					"movaps %%xmm15, %%xmm13\n\t"
					"xorps %%xmm15, %%xmm15\n\t"
					"cmpeqss %%xmm14, %%xmm15\n\t"
					"movdqa %%xmm15, %%xmm12\n\t"
					"mov $0x1, %%eax\n\t"
					"movd %%eax, %%xmm15\n\t"
					"shufps $0, %%xmm15, %%xmm15\n\t"
					"andps %%xmm15, %%xmm12\n\t"
					"xorps %%xmm15, %%xmm15\n\t"
					"movaps %0, %%xmm11\n\t"
					"movaps %1, %%xmm10\n\t"
					"mulss %%xmm12, %%xmm11\n\t"
					"mulss %%xmm14, %%xmm13\n\t"
					"addps %%xmm11, %%xmm13\n\t"
					"movaps %%xmm11, %0\n\t"
					"subps %%xmm10, %2\n\t"
					"movaps %2, %%xmm13\n\t"
					"mulss %%xmm14, %%xmm13\n\t"
					"mulss %%xmm12, %%xmm10\n\t"
					"addps %%xmm10, %%xmm13\n\t"
					"movaps %%xmm13, %1\n\t"
					"movaps %3, %%xmm13\n\t"
					"mulss %%xmm14, %%xmm13\n\t"
					"phaddd %%xmm14, %%xmm13\n\t"
					"phaddd %%xmm15, %%xmm13\n\t"
					"movd %%xmm13, %4\n\t"
					"subps %0, %5\n\t"
					"divps %1, %5\n\t"
					"subps %0, %6\n\t"
					"divps %1, %6\n\t"
					"movaps %5, %%xmm15\n\t"
					"movaps %5, %%xmm14\n\t"
					"movlhps %%xmm15, %%xmm15\n\t"
					"movhlps %%xmm14, %%xmm14\n\t"
					"maxps %%xmm14, %%xmm15\n\t"
					"movaps %%xmm15, %%xmm13\n\t"
					"unpckhps %%xmm15, %%xmm15\n\t"
					"maxps %%xmm15, %%xmm13\n\t"
					"movaps %6, %%xmm15\n\t"
					"movaps %6, %%xmm14\n\t"
					"movlhps %%xmm15, %%xmm15\n\t"
					"movhlps %%xmm14, %%xmm14\n\t"
					"minps %%xmm14, %%xmm15\n\t"
					"movaps %%xmm15, %%xmm12\n\t"
					"unpckhps %%xmm15, %%xmm15\n\t"
					"minps %%xmm15, %%xmm12\n\t"
					"cmpltps %%xmm13, %%xmm12\n\t"
					"mov $0x1, %%eax\n\t"
					"movd %%eax, %%xmm15\n\t"
					"shufps $0, %%xmm15, %%xmm15\n\t"
					"andps %%xmm15, %%xmm12\n\t"
					"movaps %%xmm12, (%7)\n\t"
					:
					:"x"(d), "x"(o), "x"(sz), "x"(mask), "r"(a0), "x"(min), "x"(max), "r"(cmp)
					:

			);

#else
			const f4v zero=__extension__(f4v){0,0,0,0};
			const u4v one=__extension__(u4v)(ivector_init){0,1,1,1};

			const f4v one_point_zero={0,1.0,1.0,1.0};
			u4v test=(u4v)f4_and((f4v)one,(f4v)f4_le(d, zero));
			f4v tmp=f4_sub(zero, d);
			d=f4_add(f4_mul(d, f4_eq((f4v)zero,(f4v) test)), (f4v) u4_mul((u4v)(f4_and((f4v)one,(f4v)f4_eq((f4v)tmp, (f4v)test))), (u4v) one_point_zero));
			tmp=f4_sub(sz, o);
			o=f4_add(f4_mul(o, (f4v)f4_eq((f4v)zero,(f4v) test)), f4_mul(tmp, (f4v)test));
			unsigned int ld[4]={1,2,4,0};

			a=(f4v)u4_mul((u4v)test,(u4v)(f4_ld((float*)ld)));
			a=(f4v)u4_hadd((u4v)a,(u4v)a);
			a=(f4v)u4_hadd((u4v)a,(u4v)a);
			__m128 t0=f4_div(f4_sub(min, o),d);
			__m128 t1=f4_div(f4_sub(max, o),d);
			tmp=f4_max(f4_movelh(t0,t0),f4_movehl(t0,t0));
			tmp=f4_max(f4_unpackhi(tmp , tmp),tmp);
			f4v t2=f4_min(f4_movelh(t1,t1),f4_movehl(t1,t1));
			t2=f4_min(f4_unpackhi((f4v)test , (f4v)test),(f4v)test);
			float tmpf[4], testf[4];
			f4_st(tmpf, tmp);
			f4_st(testf, (f4v)t2);
			f4_st(af, a);



#endif
			__builtin_expect(!cmp[1],1);
						if(!cmp[1]){
							return procsubtree(min, max, mTreeRoot, (unsigned char) a0, plane, t, hit);
						}
						return 0;
		}
		inline int procsubtree(volatile register f4v t0, volatile register f4v t1, node* ndp, int a, int plane, float* t, node** hit){
			__builtin_expect(!(ndp->children_offset), false);
			int ix;
			node nd=*ndp;
			const float z[4]={0,0,0,0};
			const float un[4]={0,1,1,1};
			const float med[4]={0,0.5,0.5,0.5};
			f4v zero=f4_ld(z);

			f4v one=f4_ld(un);
			f4v test=f4_and(one, f4_lt(t1, zero));
			float r[4];
			f4v rr=(f4v)u4_hadd(u4_hadd((u4v)test,(u4v)test),(u4v)test);
			f4_st(r,rr);
			int ret=(r[0]>0);
			__builtin_expect(ret, 0);
			f4v tm=zero;
			f4v half=f4_ld(med);
			tm=f4_mul(f4_add(t0,t1), half);
				if(!(ndp->children_offset)){
					if(ndp->n>0){
						if(*hit==NULL){
							f4_st(t, tm);
							*hit=ndp;
						}
						return 0;
					}
					return 1;
				}
				if(ret){
					return 1;
				}
#if defined(_M_SSE) && defined(_LP64)
			const 	__m128 mask11=__extension__ (__m128)(__v4si){0xffffffff,0x3020100,0x3020100,0x7060504};
			const 	__m128 mask22=__extension__ (__m128)(__v4si){0xffffffff,0xB0A0908,0xB0A0908,0x3020100};
			const 	__m128 mask33=__extension__ (__m128)(__v4si){0,1,2,4};
			 	__m128 mask44=__extension__ (__m128)(__v4si){0,1, 0, 0};

			asm volatile("movaps %1, %%xmm0;"
					"movaps %1, %%xmm2;"
					"movaps %2, %%xmm3;"
					"pshufb %3, %%xmm2;"
					"pshufb %4, %%xmm3;"
					"movaps %%xmm2, %%xmm4;"
					"movaps %%xmm2, %%xmm1;"
					"movaps %%xmm2, %%xmm7;"
					"cmpltps %%xmm0, %%xmm2;"
					"cmpltps %%xmm0, %%xmm3;"
					"cmpltps %2, %%xmm1;"
					"mov $1, %%eax;"
					"movd %%eax, %%xmm5;"
					"shufps $0, %%xmm5, %%xmm5;"
					"cmpltps %1, %%xmm4;"
					"cmpltps %2, %%xmm7;"
					"andps %%xmm5, %%xmm2;"
					"andps %%xmm5, %%xmm3;"
					"andps %%xmm5, %%xmm1;"
					"andps %%xmm5, %%xmm4;"
					"andps %%xmm5, %%xmm7;"
					"andps %%xmm1, %%xmm7;"
					"andps %%xmm1, %%xmm4;"
					"xorps %6, %%xmm4;"
					"movaps %%xmm7, %%xmm15;"
					"andps %6, %%xmm7;"
					"shufps $0x1b, %6, %6;"
					"orps %6, %%xmm1;"
					"xorps %%xmm5, %%xmm1;"
					"andps %6, %%xmm15;"
					"mulss  %5, %%xmm15;"
					"mulss %5, %%xmm7;"
					"mulss %5, %%xmm4;"
					"mulss %5, %%xmm1;"
					"xorps %%xmm0, %%xmm0;"
					"orps %%xmm0, %%xmm1;"
					"orps %%xmm0, %%xmm7;"
					"orps %%xmm0, %%xmm4;"
					"orps %%xmm0, %%xmm15;"
					"shufps $0x39, %%xmm0, %%xmm1;"
					"orps %%xmm0, %%xmm1;"
					"shufps $0x39, %%xmm1, %%xmm2;"
					"orps %%xmm1, %%xmm2;"
					"movd %%xmm2, %0;"
					:"=r"(ix)
					:"x"(t0),"x"(tm), "x"(mask11), "x"(mask22), "x"(mask33), "x"(mask44)
				);
#else
			 float t0f[4], tmf[4];

			       f4_st(t0f,t0);

			       f4_st(tmf,tm);
			       ix=0;
			    int x=(t0f[1] < t0f[0]);
			                        int xy= x*(t0f[2] < t0f[1]);
			                               ix |= 2*(tmf[1] < t0f[0])*xy;
			                               ix |= 1*(tmf[2] < t0f[0])*xy;
			                   int xz=x*(tmf[2] < t0f[1]);
			                               ix |= 4*(tmf[0] < t0f[1])*xz;
		                          ix |= 1*(tmf[2] < t0f[0])*xz;



			                    // XY plane

			                ix |= 4*(tmf[0] < t0f[2])*!x;
			                ix |= (tmf[1] < t0f[2])*2*!x;
#endif


			__builtin_expect(ix<8, 1);
			while(ix<8){
#ifdef SSE42
				register const int z=(ix&1)*0xffffffff, y=(ix&2)*0xffffffff, x=(ix&4)*0xffffffff;
									__m128 mask=__extension__ (__m128)(__v4si){0,z,y,x};
									t0=_mm_blendv_ps(t0, tm, mask);
									__m128 mask=__extension__ (__m128)(__v4si){0,!z.!y,!x};
									tm=_mm_blendv_ps(tm, t0, mask);
#else
					register const int z=(ix&1), y=(ix&2), x=(ix&4);
					u4v mask=__extension__(u4v)(ivector_init){0xffffffff,0xffffffff*!z+z*0xB0A0908,y*0x7060504+0xffffffff*!y+1,x*0x3020100+0xffffffff*!x};
					u4v mask1=__extension__ (u4v)(ivector_init){0xffffffff,z*0xffffffff+0xB0A0908*!z,y*0xffffffff+0x7060504*!y,x*0xffffffff+0x3020100*!x};
					t0=f4_add((f4v)u4_pshufb( (u4v)t0, (u4v)mask), (f4v)u4_pshufb( (u4v)tm,  (u4v)mask1));
					tm=f4_add((f4v)u4_pshufb( (u4v)t0,  (u4v)mask1), (f4v)u4_pshufb( (u4v)tm,  (u4v)mask));

#endif
					/*
					asm volatile(
							"movaps %1, %%xmm14\n\t"
							"movaps %0, %%xmm0\n\t"
							"xorps %%xmm15, %%xmm15\n\t"
							"cmpeqss %%xmm15, %%xmm0 \n\t"
							"blendvps %%xmm14, %2 \n\t"
							:
							: "Yz"(mask), "x" (t0), "x"(tm)
					);*/
					int newhit=procsubtree(t0, tm, &(ndp[nd.children_offset+(ix^a)]), a, plane, t, hit);
					__builtin_expect(!newhit, 1);
					if(!newhit) return 0;

					const u4v mask2=__extension__ (u4v)(ivector_init){0xffffffff,0xffffffff,0xB0A0908,0x7060504};

					const u4v mask3=__extension__ (u4v)(ivector_init){1,1,1,1};
					f4v xmm0=(f4v)u4_pshufb((u4v)t0, (u4v)mask2);
					xmm0=f4_lt(t0, xmm0);
					xmm0=(f4v) f4_and((f4v)xmm0, (f4v)mask3);
					xmm0=(f4v) u4_hadd((u4v)xmm0, (u4v)xmm0);
					xmm0=(f4v) u4_hadd((u4v)xmm0, (u4v)xmm0);
					int pln[4];
					f4_st((float*) pln,xmm0);
					ix=newix[pln[0]+ix*3];
					__builtin_expect(ix<8, 1);
			}
			return 1;
		}

	};