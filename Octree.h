
struct OctreeBV{
	int size_x;
	int size_y;
	int* objects; //c-style cast to this from below == dirty fun :D
};
template<unsigned char levels=9, unsigned int max_items=5, int siz=32*1024*1024> class Octree{
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

		__m128 c=_mm_load_ps(coords);
		__m128 center=_mm_set1_ps(0.5);
		for(unsigned char lvl=0; lvl<levels; ++lvl){
			if(!(nd->children_offset)){
							break;
			}
			unsigned ix;
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
		inline int rayparam(volatile register __m128 min,  volatile register __m128 max,  volatile register __m128 sz, volatile register __m128 d, volatile register  __m128 o, volatile register __m128 a, unsigned char plane, float* t, node** hit){
			unsigned cmp[4]={0,0,0,0};
			unsigned a0=0;
			volatile register __m128 mask=__extension__ (__m128)(__v4sf){ 0.,4.,2.,1.};
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
			/*
			__m128 zero=_mm_setzero_ps();
			__m128 test=_mm_andnot_ps(zero,_mm_cmplt_ps(d, zero));
			__m128 tmp=_mm_sub_ps(zero, d);
			d=_mm_add_ps(_mm_mul_ps(d, _mm_cmpeq_ps(zero, test)), _mm_mul_ps(tmp, test));
			tmp=_mm_sub_ps(sz, o);
			o=_mm_add_ps(_mm_mul_ps(o, _mm_cmpeq_ps(zero, test)), _mm_mul_ps(tmp, test));
			a=dpps_instr(test, _mm_set_ps(1,2,4,0), 0xFF);
			__m128 t0=_mm_div_ps(_mm_sub_ps(min, o),d);
			__m128 t1=_mm_div_ps(_mm_sub_ps(max, o),d);
			tmp=_mm_max_ps(_mm_movelh_ps(t0,t0),_mm_movehl_ps(t0,t0));
			tmp=_mm_max_ps(_mm_unpackhi_ps(tmp , tmp),tmp);
			test=_mm_min_ps(_mm_movelh_ps(t1,t1),_mm_movehl_ps(t1,t1));
			test=_mm_min_ps(_mm_unpackhi_ps(test , test),test);
			float tmpf[4], testf[4], af[4];
			_mm_store_ps(tmpf, tmp);
			_mm_store_ps(testf, test);
			_mm_store_ps(af, a);
			*/
			__builtin_expect(!cmp[1],1);
			if(!cmp[1]){
				return procsubtree(min, max, mTreeRoot, (unsigned char) a0, plane, t, hit);
			}
			return 0;
		}

		inline int procsubtree(volatile register __m128 t0, volatile register __m128 t1, node* ndp, int a, int plane, float* t, node** hit){
			__builtin_expect(!(ndp->children_offset), false);
			int ix;
			node nd=*ndp;
			__m128 zero=_mm_setzero_ps();
			__m128 test=_mm_and_ps(_mm_set1_ps(1.0), _mm_cmplt_ps(t1, zero));
			int r;
			asm volatile(
										"phaddd %1, %1;"
										"phaddd %1, %1;"
										"movd %1, %0 \n \t"
										: "=r"(r)
										: "x" (test)
										:
								);

			int ret=(r>0);
			__builtin_expect(ret, 0);
			__m128 tm=_mm_setzero_ps();
			__asm volatile("mov $0x3840000, %%eax;"
					" movd %%eax, %%xmm0; "
					"movaps %1, %2; "
					"addps %0, %2; "
					"mulps  %%xmm0, %2;"
					:
					:"x"(t0),"x"(t1),"x"(tm));


				if(!(ndp->children_offset)){
					if(ndp->n>0){
						if(*hit==NULL){
							_mm_store_ps(t, tm);
							*hit=ndp;
						}
						return 0;
					}
					return 1;
				}
				if(ret){
					return 1;
				}

				__m128 mask11=__extension__ (__m128)(__v4si){3,0,0,1};

				__m128 mask22=__extension__ (__m128)(__v4si){3,2,2,0};
				__m128 mask33=__extension__ (__m128)(__v4si){0,1,2,4};
				__m128 mask44=__extension__ (__m128)(__v4si){0,1, 0, 0};
			asm volatile("movaps %1, %%xmm0;"
					"movaps %1, %%xmm2;"
					"movaps %2, %%xmm3;"
					"pshufb %3, %%xmm2;"
					"pshufb %4, %%xmm3;"
					"movaps %%xmm2, %%xmm4;"
					"movaps %%xmm2, %%xmm1;"
					"movaps %%xmm2, %%xmm7;"
					"cmpltps %%xmm1, %%xmm2;"
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


			__builtin_expect(ix<8, 1);
			while(ix<8){
					register const int z=(ix&1)*0xffffffff, y=(ix&2)*0xffffffff, x=(ix&4)*0xffffffff;
					__m128 mask=__extension__ (__m128)(__v4si){0,z,y,x};
					asm volatile(
							"movaps %2, %%xmm13\n\t"
							"movaps %0, %%xmm0\n\t"
							"blendvps %%xmm13, %1 \n\t"
							:
							: "Yz"(mask), "x" (t0), "x"(tm)
						);
					asm volatile(
							"movaps %1, %%xmm14\n\t"
							"movaps %0, %%xmm0\n\t"
							"xorps %%xmm15, %%xmm15\n\t"
							"cmpeqss %%xmm15, %%xmm0 \n\t"
							"blendvps %%xmm14, %2 \n\t"
							:
							: "Yz"(mask), "x" (t0), "x"(tm)
					);
					int newhit=procsubtree(t0, tm, &(ndp[nd.children_offset+(ix^a)]), a, plane, t, hit);
					__builtin_expect(!newhit, 1);
					if(!newhit) return 0;

					volatile register __m128 mask2=__extension__ (__m128)(__v4si){0xff,0xff,2,1};

					__asm("movaps %1, %%xmm0; "
							"pshufb %2, %%xmm0;"
							" cmpltps %1, %%xmm0; "
							"mov $0x3f800000, %%eax;"
							" movd %%eax, %2; "
							"andps %2, %%xmm0; "
							" phaddd %%xmm0, %%xmm0; "
							"phaddd %%xmm0, %%xmm0; "
							"movd %%xmm0, %0"
							:"=r"(plane)
							:"x"(t0),"x"(mask2)
							 );

					ix=newix[plane+ix*3];
					__builtin_expect(ix<8, 1);
			}
			return 1;
		}

	};
