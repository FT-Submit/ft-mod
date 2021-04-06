#include <iostream>
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include <random>
#include <cstring>
#include <algorithm>
#include <chrono>

#define CACHELINE 64
#define FPCACHE 16
#define GIGA 1000000000
#define FULL_TP false
#define RELAX false

#define MODE 3

#define L1_SIZE 32000
#define L2_SIZE 1000000
#define FREQ 2.2

uint64_t n = 1000;
uint64_t m = 300;
uint64_t mv = 1;
uint64_t padm = m + FPCACHE - (m & (FPCACHE - 1));
uint64_t losses = 6;

uint64_t rounds = 50;
uint64_t warmupl = 2000;
uint64_t globalctr = 0;

inline void do_hidden(float *a, float *b);
inline void do_hidden(float *a, float *x, std::vector<int64_t> rand);
inline void do_hidden_v(__m512 *av, __m512 *xv);
inline void do_hidden_v_fulltp(__m512 *av, __m512 *xv);
inline void do_hidden_v(__m512 *av, __m512 *xv, int64_t* rand);

inline void save_hidden_v(__m512 *av, __m512 *xv, int64_t *rand);
inline void save_hidden_v(__m512 *av, __m512 *xv);
inline void save_hidden_v_fulltp(__m512 *av, __m512 *xv);

inline void do_loss_v(__m512 *av, __m512 *xv, __m512 *gv, int64_t *rand);
inline void do_loss_v(__m512 *av, __m512 *xv, __m512 *gv);
inline void do_loss_v_fulltp(__m512 *av, __m512 *xv, __m512 *gv);

inline void test_reduce_add(__m512 *kv, float *r) {
  for (int64_t i = 0; i < mv; ++i) {
    r[0] += _mm512_reduce_add_ps(kv[i]);
  }
}

//UP TO HERE

inline void do_all(__m512 *av, __m512 *bv, __m512 *xv, __m512 *gv);
inline void do_all(__m512 *av, __m512 *bv, __m512 *xv, __m512 *gv, int64_t *rand);

typedef struct timespec clocktime;

inline int64_t getTimeDifference(clocktime *start, clocktime *end);
inline int getTime(clocktime *time);

int main(int argc, char *argv[]) {

  /*if (argc == 3) {
    n = std::atoi(argv[1]);
    m = std::atoi(argv[2]);
  }*/

  //std::cout << mkl_get_max_threads() << std::endl;

 // mkl_set_num_threads(1);
  uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t ns, ls;

  //std::cout << "size,block4,block5,block6\n";
  std::cout << "size,,avg_4,avg_5,avg_6,,med_4,med_5,med_6\n";
  std::vector<uint64_t> res1(rounds);
  std::vector<uint64_t> res2(rounds);
  std::vector<uint64_t> res3(rounds);
  std::vector<uint64_t> res4(rounds);
  std::vector<uint64_t> idxs = {1, 2, 4};
  int32_t ngval = 7;
  while (ngval < 300) {
    idxs.push_back(ngval);
    ngval += 4;
  }
   //~ = {1/**/, 2, 4, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99,
                                //~ 103, 107, /*/, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000/**/};
  std::vector<uint64_t> idxsm = {100, 300, 500, 1000};

  std::cout << "mem,vsize,ngrams,repeated,sequential,sequential-array,random\n";

  for (uint64_t km = 0; km < idxsm.size(); ++km) {
  for (uint64_t k = 0; k < idxs.size(); ++k) {
    n = idxs[k];
    m = idxsm[km];
    losses = k + 2;
    ns = n;//n;
    ls = 1000;

    if ((n + 2) * m * 4 > L2_SIZE) {
      //continue;
      std::cout << "MM,";
    } else if ((n + 2) * m * 4 > L1_SIZE) {
      //continue;
      std::cout << "L2,";
    } else {
      std::cout << "L1,";
    }
    //    std::cout << n * m * 4 << ",";
    mv = (m + FPCACHE - 1) / FPCACHE;
    //n = idxs[k];
    //m = idxs[k];
    padm = (m + FPCACHE - 1) / FPCACHE * FPCACHE;//m + CACHELINE - (m & (CACHELINE - 1));

    uint64_t tot1 = 0;
    uint64_t tot2 = 0;
    uint64_t tot3 = 0;
    uint64_t tot4 = 0;

    #if MODE == 3
    uint64_t rands = ls;
    #else
    uint64_t rands = ns;
    #endif

    int64_t* rand;
    int64_t* sequ;
    int rr = posix_memalign((inline void**)&rand, CACHELINE, rands * sizeof(int64_t));
    int rr2 = posix_memalign((inline void**)&sequ, CACHELINE, rands * sizeof(int64_t));
    for (int64_t i = 0; i < rands; ++i) {
      rand[i] = i;
      sequ[i] = i;
    }
    std::shuffle(rand, rand + rands, std::default_random_engine(seed));

    for (uint64_t r = 0; r < rounds; ++r) {

      float *a1, *x1, *a2, *x2, *b1, *b2;
      float *g1, *g2;

      clocktime t11, t12, t21, t22, t31, t32, t41, t42;

      std::default_random_engine generator;
      std::uniform_real_distribution<float> distribution(-1.0, 1.0);

      __m512 *av;
      __m512 *bv;
      __m512 *xv;
      __m512 *gv;

      int r1 = posix_memalign((inline void**)&a1, CACHELINE, ns * padm * sizeof(float));
      int rb1 = posix_memalign((inline void**)&b1, CACHELINE, ls * padm * sizeof(float));
      int r2 = posix_memalign((inline void**)&x1, CACHELINE, padm * sizeof(float));
      //int r3 = posix_memalign((inline void**)&a2, CACHELINE, ns * padm * sizeof(float));
      //int rb2 = posix_memalign((inline void**)&b2, CACHELINE, ns * padm * sizeof(float));
      //int r5 = posix_memalign((inline void**)&x2, CACHELINE, padm * sizeof(float));
      int r7 = posix_memalign((inline void**)&g1, CACHELINE, padm * sizeof(float));
      //int r8 = posix_memalign((inline void**)&g2, CACHELINE, padm * sizeof(float));

      for (uint64_t i = 0; i < ns * padm; ++i)
        a1[i] = distribution(generator);
      for (uint64_t i = 0; i < ls * padm; ++i)
        b1[i] = distribution(generator);
      for (uint64_t i = 0; i < m; ++i)
        x1[i] = distribution(generator);
      for (uint64_t i = m; i < padm; ++i)
        x1[i] = 0;
      for (uint64_t i = 0; i < padm; ++i)
        g1[i] = 0;


      int rv1 = posix_memalign((inline void**)&av, CACHELINE, ns * mv * sizeof(__m512));
      int rvb = posix_memalign((inline void**)&bv, CACHELINE, ls * mv * sizeof(__m512));
      int rv2 = posix_memalign((inline void**)&xv, CACHELINE, mv * sizeof(__m512));
      int rv3 = posix_memalign((inline void**)&gv, CACHELINE, mv * sizeof(__m512));

      /*std::memcpy(a2, a1, ns * padm * sizeof(float));
      std::memcpy(b2, b1, ns * padm * sizeof(float));
      std::memcpy(x2, x1, padm * sizeof(float));
      std::memcpy(g2, g1, padm * sizeof(float));*/

      //for (uint64_t r = 0; r < rounds; ++r) {

      for (uint64_t i = 0; i < ns * mv; ++i) {
        av[i] = _mm512_loadu_ps(&a1[i * 16]);
      }
      for (uint64_t i = 0; i < ls * mv; ++i) {
        bv[i] = _mm512_loadu_ps(&b1[i * 16]);
      }
      for (uint64_t i = 0; i < mv; ++i) {
        xv[i] = _mm512_loadu_ps(&x1[i * 16]);
        gv[i] = _mm512_loadu_ps(&g1[i * 16]);
      }

      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v_fulltp(av, xv);
        #elif MODE == 2
        save_hidden_v_fulltp(av, xv);
        #else
        do_loss_v_fulltp(bv, xv, gv);
        #endif
      }

      getTime(&t11);
      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v_fulltp(av, xv);
        #elif MODE == 2
        save_hidden_v_fulltp(av, xv);
        #else
        do_loss_v_fulltp(bv, xv, gv);
        #endif
      }
      getTime(&t12);

      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv);
        #elif MODE == 2
        save_hidden_v(av, xv);
        #else
        do_loss_v(bv, xv, gv);
        #endif
      }

      getTime(&t21);
      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv);
        #elif MODE == 2
        save_hidden_v(av, xv);
        #else
        do_loss_v(bv, xv, gv);
        #endif
      }
      getTime(&t22);

      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv, sequ);
        #elif MODE == 2
        save_hidden_v(av, xv, sequ);
        #else
        do_loss_v(bv, xv, gv, sequ);
        #endif
      }

      getTime(&t31);
      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv, sequ);
        #elif MODE == 2
        save_hidden_v(av, xv, sequ);
        #else
        do_loss_v(bv, xv, gv, sequ);
        #endif
      }
      getTime(&t32);

      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv, rand);
        #elif MODE == 2
        save_hidden_v(av, xv, rand);
        #else
        do_loss_v(bv, xv, gv, rand);
        #endif
      }

      getTime(&t41);
      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) {
        #if MODE == 1
        do_hidden_v(av, xv, rand);
        #elif MODE == 2
        save_hidden_v(av, xv, rand);
        #else
        do_loss_v(bv, xv, gv, rand);
        #endif
      }
      getTime(&t42);

      //free(kv);
      //free(kres);

      /*std::memcpy(a2, a1, n * padm * sizeof(float));
      std::memcpy(x2, x1, padm * sizeof(float));
      getTime(&t31);
      do_hidden(a2, x2);
      getTime(&t32);

      for (uint64_t warmup = 0; warmup < warmupl; ++warmup) do_hidden(a2, x2);

      getTime(&t41);
      do_hidden(a2, x2);
      getTime(&t42);*/

      res1[r] = getTimeDifference(&t11, &t12);
      res2[r] = getTimeDifference(&t21, &t22);
      res3[r] = getTimeDifference(&t31, &t32);
      res4[r] = getTimeDifference(&t41, &t42);
      tot1 += getTimeDifference(&t11, &t12);
      tot2 += getTimeDifference(&t21, &t22);
      tot3 += getTimeDifference(&t31, &t32);
      tot4 += getTimeDifference(&t41, &t42);

      free(a1);
      free(b1);
      free(x1);
      free(g1);
      //free(a2);
      //free(b2);
      //free(x2);
      //free(g2);
      free(av);
      free(bv);
      free(xv);
      free(gv);
    }

    free(rand);
    free(sequ);

    std::sort(res1.begin(), res1.end());
    std::sort(res2.begin(), res2.end());
    std::sort(res3.begin(), res3.end());
    std::sort(res4.begin(), res4.end());

    int64_t c = 0; //dotp reduction cost
    if (m == 300) c = 18 * 16 + 15;
    else if (m == 100) c = 6 * 16 + 15;
    else c = 11 * 16 + 15;
    #if MODE == 3
    int64_t flops = losses * (padm * 6 - 1);
    #else
    int64_t flops = n * padm;
    #endif
    //int64_t flops = n * padm * 2 + losses * (padm * 6 - 1);
    //int64_t flops = losses * (padm * 6 - 1);
    float freq = FREQ;

    #if MODE == 3
    int32_t printv = losses - 1;
    #else
    int32_t printv = n;
    #endif

    std::cout << m << "," << printv <<
                /*",," << tot1 / rounds << "," <<
                  res1[rounds/2] << "," << (tot1 * 1.0 / (rounds * warmupl)) * freq <<*/ "," << flops * 1.0 / ((tot1 * 1.0 / (rounds * warmupl)) * freq) <<
                /*",," << tot2 / rounds << "," <<
                  res2[rounds/2] << "," << (tot2 * 1.0 / (rounds * warmupl)) * freq <<*/ "," << flops * 1.0 / ((tot2 * 1.0 / (rounds * warmupl)) * freq) <<
                /*",," << tot3 / rounds << "," <<
                  res3[rounds/2] << "," << (tot3 * 1.0 / (rounds * warmupl)) * freq <<*/ "," << flops * 1.0 / ((tot3 * 1.0 / (rounds * warmupl)) * freq) <<
                /*",," << tot4 / rounds << "," <<
                  res4[rounds/2] << "," << (tot4 * 1.0 / (rounds * warmupl)) * freq <<*/ "," << flops * 1.0 / ((tot4 * 1.0 / (rounds * warmupl)) * freq) <<
    std::endl;
  }
  }

  return 0;
}

inline void do_all(__m512 *av, __m512 *bv, __m512 *xv, __m512 *gv) {
  do_hidden_v(av, xv);
  do_loss_v(bv, xv, gv);
  save_hidden_v(av, gv);
}

inline void do_all(__m512 *av, __m512 *bv, __m512 *xv, __m512 *gv, int64_t *rand) {
  do_hidden_v(av, xv, rand);
  do_loss_v(bv, xv, gv, rand);
  save_hidden_v(av, gv, rand);
}

inline void do_hidden_v_fulltp(__m512 *av, __m512 *xv) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  vvec1 = _mm512_setzero_ps();
  vvec2 = _mm512_setzero_ps();
  vvec3 = _mm512_setzero_ps();
  vvec4 = _mm512_setzero_ps();
  vvec5 = _mm512_setzero_ps();
  vvec6 = _mm512_setzero_ps();
  vvec7 = _mm512_setzero_ps();
  vvec8 = _mm512_setzero_ps();
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = _mm512_setzero_ps();
    vvec2 = _mm512_setzero_ps();
    vvec3 = _mm512_setzero_ps();
    vvec4 = _mm512_setzero_ps();
    vvec5 = _mm512_setzero_ps();
    vvec6 = _mm512_setzero_ps();
    vvec7 = _mm512_setzero_ps();
    vvec8 = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = 0;
      vvec1 = _mm512_add_ps(av[i * mv + j], vvec1);
      vvec2 = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      vvec3 = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      vvec4 = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      vvec5 = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      vvec6 = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      vvec7 = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      vvec8 = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
    xv[j] = vvec1;
    xv[j + 1] = vvec2;
    xv[j + 2] = vvec3;
    xv[j + 3] = vvec4;
    xv[j + 4] = vvec5;
    xv[j + 5] = vvec6;
    xv[j + 6] = vvec7;
    xv[j + 7] = vvec8;
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = 0;
      vvec = _mm512_add_ps(av[i * mv + j], vvec);
    }
    xv[j] = vvec;
  }
}

inline void do_hidden_v(__m512 *av, __m512 *xv) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  vvec1 = _mm512_setzero_ps();
  vvec2 = _mm512_setzero_ps();
  vvec3 = _mm512_setzero_ps();
  vvec4 = _mm512_setzero_ps();
  vvec5 = _mm512_setzero_ps();
  vvec6 = _mm512_setzero_ps();
  vvec7 = _mm512_setzero_ps();
  vvec8 = _mm512_setzero_ps();
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = _mm512_setzero_ps();
    vvec2 = _mm512_setzero_ps();
    vvec3 = _mm512_setzero_ps();
    vvec4 = _mm512_setzero_ps();
    vvec5 = _mm512_setzero_ps();
    vvec6 = _mm512_setzero_ps();
    vvec7 = _mm512_setzero_ps();
    vvec8 = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = iv;
      vvec1 = _mm512_add_ps(av[i * mv + j], vvec1);
      vvec2 = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      vvec3 = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      vvec4 = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      vvec5 = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      vvec6 = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      vvec7 = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      vvec8 = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
    xv[j] = vvec1;
    xv[j + 1] = vvec2;
    xv[j + 2] = vvec3;
    xv[j + 3] = vvec4;
    xv[j + 4] = vvec5;
    xv[j + 5] = vvec6;
    xv[j + 6] = vvec7;
    xv[j + 7] = vvec8;
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = iv;
      vvec = _mm512_add_ps(av[i * mv + j], vvec);
    }
    xv[j] = vvec;
  }
}

inline void do_hidden_v(__m512 *av, __m512 *xv, int64_t *rand) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  vvec1 = _mm512_setzero_ps();
  vvec2 = _mm512_setzero_ps();
  vvec3 = _mm512_setzero_ps();
  vvec4 = _mm512_setzero_ps();
  vvec5 = _mm512_setzero_ps();
  vvec6 = _mm512_setzero_ps();
  vvec7 = _mm512_setzero_ps();
  vvec8 = _mm512_setzero_ps();
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = _mm512_setzero_ps();
    vvec2 = _mm512_setzero_ps();
    vvec3 = _mm512_setzero_ps();
    vvec4 = _mm512_setzero_ps();
    vvec5 = _mm512_setzero_ps();
    vvec6 = _mm512_setzero_ps();
    vvec7 = _mm512_setzero_ps();
    vvec8 = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = rand[iv];
      vvec1 = _mm512_add_ps(av[i * mv + j], vvec1);
      vvec2 = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      vvec3 = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      vvec4 = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      vvec5 = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      vvec6 = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      vvec7 = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      vvec8 = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
    xv[j] = vvec1;
    xv[j + 1] = vvec2;
    xv[j + 2] = vvec3;
    xv[j + 3] = vvec4;
    xv[j + 4] = vvec5;
    xv[j + 5] = vvec6;
    xv[j + 6] = vvec7;
    xv[j + 7] = vvec8;
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = _mm512_setzero_ps();
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = rand[iv];
      vvec = _mm512_add_ps(av[i * mv + j], vvec);
    }
    xv[j] = vvec;
  }
}

inline void save_hidden_v(__m512 *av, __m512 *xv, int64_t *rand) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = xv[j + 0];
    vvec2 = xv[j + 1];
    vvec3 = xv[j + 2];
    vvec4 = xv[j + 3];
    vvec5 = xv[j + 4];
    vvec6 = xv[j + 5];
    vvec7 = xv[j + 6];
    vvec8 = xv[j + 7];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = rand[iv];
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec1);
      av[i * mv + j + 1] = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      av[i * mv + j + 2] = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      av[i * mv + j + 3] = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      av[i * mv + j + 4] = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      av[i * mv + j + 5] = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      av[i * mv + j + 6] = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      av[i * mv + j + 7] = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = xv[j + 0];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = rand[iv];
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec);
    }
  }
}

inline void save_hidden_v_fulltp(__m512 *av, __m512 *xv) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = xv[j + 0];
    vvec2 = xv[j + 1];
    vvec3 = xv[j + 2];
    vvec4 = xv[j + 3];
    vvec5 = xv[j + 4];
    vvec6 = xv[j + 5];
    vvec7 = xv[j + 6];
    vvec8 = xv[j + 7];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = 0;
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec1);
      av[i * mv + j + 1] = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      av[i * mv + j + 2] = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      av[i * mv + j + 3] = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      av[i * mv + j + 4] = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      av[i * mv + j + 5] = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      av[i * mv + j + 6] = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      av[i * mv + j + 7] = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = xv[j + 0];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = 0;
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec);
    }
  }
}

inline void save_hidden_v(__m512 *av, __m512 *xv) {
  __m512 vvec;
  __m512 vvec1, vvec2, vvec3, vvec4, vvec5, vvec6;
  __m512 vvec7, vvec8;
  int64_t m128 = mv - (mv % 8);
  for (int64_t j = 0; j < m128; j += 8) {
    vvec1 = xv[j + 0];
    vvec2 = xv[j + 1];
    vvec3 = xv[j + 2];
    vvec4 = xv[j + 3];
    vvec5 = xv[j + 4];
    vvec6 = xv[j + 5];
    vvec7 = xv[j + 6];
    vvec8 = xv[j + 7];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = iv;
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec1);
      av[i * mv + j + 1] = _mm512_add_ps(av[i * mv + j + 1], vvec2);
      av[i * mv + j + 2] = _mm512_add_ps(av[i * mv + j + 2], vvec3);
      av[i * mv + j + 3] = _mm512_add_ps(av[i * mv + j + 3], vvec4);
      av[i * mv + j + 4] = _mm512_add_ps(av[i * mv + j + 4], vvec5);
      av[i * mv + j + 5] = _mm512_add_ps(av[i * mv + j + 5], vvec6);
      av[i * mv + j + 6] = _mm512_add_ps(av[i * mv + j + 6], vvec7);
      av[i * mv + j + 7] = _mm512_add_ps(av[i * mv + j + 7], vvec8);
    }
  }
  for (int64_t j = m128; j < mv; j += 1) {
    vvec = xv[j + 0];
    for (int64_t iv = 0; iv < n; ++iv) {
      int64_t i = iv;
      av[i * mv + j] = _mm512_add_ps(av[i * mv + j], vvec);
    }
  }
}

inline void do_loss_v(__m512 *av, __m512 *xv, __m512 *gv, int64_t *rand) {
  for (int64_t iv = 0; iv < losses; ++iv) {
    int64_t i = rand[iv];

    //DOT
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m512 acc8 = _mm512_setzero_ps();
    int64_t m128 = mv - (mv % 8);
    for (int64_t j = 0; j < m128; j += 8) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
      acc2 = _mm512_fmadd_ps(av[i * mv + j + 1], xv[j + 1], acc2);
      acc3 = _mm512_fmadd_ps(av[i * mv + j + 2], xv[j + 2], acc3);
      acc4 = _mm512_fmadd_ps(av[i * mv + j + 3], xv[j + 3], acc4);
      acc5 = _mm512_fmadd_ps(av[i * mv + j + 4], xv[j + 4], acc5);
      acc6 = _mm512_fmadd_ps(av[i * mv + j + 5], xv[j + 5], acc6);
      acc7 = _mm512_fmadd_ps(av[i * mv + j + 6], xv[j + 6], acc7);
      acc8 = _mm512_fmadd_ps(av[i * mv + j + 7], xv[j + 7], acc8);
    }
    for (int64_t j = m128; j < mv; j += 1) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
    }
    acc1 = _mm512_add_ps(acc1, acc2);
    acc3 = _mm512_add_ps(acc3, acc4);
    acc5 = _mm512_add_ps(acc5, acc6);
    acc7 = _mm512_add_ps(acc7, acc8);
    acc1 = _mm512_add_ps(acc1, acc3);
    acc5 = _mm512_add_ps(acc5, acc7);
    acc1 = _mm512_add_ps(acc1, acc5);
    float score = _mm512_reduce_add_ps(acc1);

    //BACKPROP
    __m512 avec = _mm512_set1_ps(score);
    __m512  dvec, gvec, hvec, rvec;
    for (int64_t j = 0; j < mv; ++j) {
      dvec = av[i * mv + j];
      gvec = gv[j];
      hvec = xv[j];
      rvec = _mm512_fmadd_ps(avec, dvec, gvec);
      av[i * mv + j] = _mm512_fmadd_ps(hvec, avec, dvec);
      gv[j] = rvec;
    }
  }
}

inline void do_loss_v_fulltp(__m512 *av, __m512 *xv, __m512 *gv) {
  for (int64_t iv = 0; iv < losses; ++iv) {
    int64_t i = 0;

    //DOT
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m512 acc8 = _mm512_setzero_ps();
    int64_t m128 = mv - (mv % 8);
    for (int64_t j = 0; j < m128; j += 8) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
      acc2 = _mm512_fmadd_ps(av[i * mv + j + 1], xv[j + 1], acc2);
      acc3 = _mm512_fmadd_ps(av[i * mv + j + 2], xv[j + 2], acc3);
      acc4 = _mm512_fmadd_ps(av[i * mv + j + 3], xv[j + 3], acc4);
      acc5 = _mm512_fmadd_ps(av[i * mv + j + 4], xv[j + 4], acc5);
      acc6 = _mm512_fmadd_ps(av[i * mv + j + 5], xv[j + 5], acc6);
      acc7 = _mm512_fmadd_ps(av[i * mv + j + 6], xv[j + 6], acc7);
      acc8 = _mm512_fmadd_ps(av[i * mv + j + 7], xv[j + 7], acc8);
    }
    for (int64_t j = m128; j < mv; j += 1) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
    }
    acc1 = _mm512_add_ps(acc1, acc2);
    acc3 = _mm512_add_ps(acc3, acc4);
    acc5 = _mm512_add_ps(acc5, acc6);
    acc7 = _mm512_add_ps(acc7, acc8);
    acc1 = _mm512_add_ps(acc1, acc3);
    acc5 = _mm512_add_ps(acc5, acc7);
    acc1 = _mm512_add_ps(acc1, acc5);
    float score = _mm512_reduce_add_ps(acc1);

    //BACKPROP
    __m512 avec = _mm512_set1_ps(score);
    __m512  dvec, gvec, hvec, rvec;
    for (int64_t j = 0; j < mv; ++j) {
      dvec = av[i * mv + j];
      gvec = gv[j];
      hvec = xv[j];
      rvec = _mm512_fmadd_ps(avec, dvec, gvec);
      av[i * mv + j] = _mm512_fmadd_ps(hvec, avec, dvec);
      gv[j] = rvec;
    }
  }
}

inline void do_loss_v(__m512 *av, __m512 *xv, __m512 *gv) {
  for (int64_t iv = 0; iv < losses; ++iv) {
    int64_t i = iv;

    //DOT
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    __m512 acc8 = _mm512_setzero_ps();
    int64_t m128 = mv - (mv % 8);
    for (int64_t j = 0; j < m128; j += 8) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
      acc2 = _mm512_fmadd_ps(av[i * mv + j + 1], xv[j + 1], acc2);
      acc3 = _mm512_fmadd_ps(av[i * mv + j + 2], xv[j + 2], acc3);
      acc4 = _mm512_fmadd_ps(av[i * mv + j + 3], xv[j + 3], acc4);
      acc5 = _mm512_fmadd_ps(av[i * mv + j + 4], xv[j + 4], acc5);
      acc6 = _mm512_fmadd_ps(av[i * mv + j + 5], xv[j + 5], acc6);
      acc7 = _mm512_fmadd_ps(av[i * mv + j + 6], xv[j + 6], acc7);
      acc8 = _mm512_fmadd_ps(av[i * mv + j + 7], xv[j + 7], acc8);
    }
    for (int64_t j = m128; j < mv; j += 1) {
      acc1 = _mm512_fmadd_ps(av[i * mv + j], xv[j], acc1);
    }
    acc1 = _mm512_add_ps(acc1, acc2);
    acc3 = _mm512_add_ps(acc3, acc4);
    acc5 = _mm512_add_ps(acc5, acc6);
    acc7 = _mm512_add_ps(acc7, acc8);
    acc1 = _mm512_add_ps(acc1, acc3);
    acc5 = _mm512_add_ps(acc5, acc7);
    acc1 = _mm512_add_ps(acc1, acc5);
    float score = _mm512_reduce_add_ps(acc1);

    //BACKPROP
    __m512 avec = _mm512_set1_ps(score);
    __m512  dvec, gvec, hvec, rvec;
    for (int64_t j = 0; j < mv; ++j) {
      dvec = av[i * mv + j];
      gvec = gv[j];
      hvec = xv[j];
      rvec = _mm512_fmadd_ps(avec, dvec, gvec);
      av[i * mv + j] = _mm512_fmadd_ps(hvec, avec, dvec);
      gv[j] = rvec;
    }
  }
}

inline int64_t getTimeDifference(clocktime *start, clocktime *end){
    int64_t secondsDiff = (int64_t)(end->tv_sec) - (int64_t)(start->tv_sec);
    int64_t endNano = end->tv_nsec;
    int64_t startNano = start->tv_nsec;
    if(startNano <= endNano)
        return secondsDiff * GIGA + endNano - startNano;
    return (secondsDiff - 1) * GIGA + GIGA - (startNano - endNano);
}

inline int getTime(clocktime *time){
    return clock_gettime(CLOCK_REALTIME, time);
}