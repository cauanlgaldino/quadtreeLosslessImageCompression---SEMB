#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>


#define MAX_W  600
#define MAX_H  600
#define MAX_STACK_NODES 1024

#define DEFAULT_TAU2      400u
#define DEFAULT_S_MIN     4u
#define DEFAULT_DEPTH_MAX 5u

// ======================= Timing helper =======================
static double now_ms(void){
#if defined(CLOCK_MONOTONIC)
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
#else
    return 0.0;
#endif
}

// ======================= Bitstream (writer/reader) =======================
typedef struct {
    uint8_t* buf;
    uint32_t bitpos;   // posição em bits
    uint32_t capBytes; // capacidade em bytes
} BitWriter;

typedef struct {
    const uint8_t* buf;
    uint32_t bitpos;
    uint32_t sizeBytes;
} BitReader;

static void bw_init(BitWriter* bw, uint8_t* backing, uint32_t capBytes){
    bw->buf = backing; bw->capBytes = capBytes; bw->bitpos = 0;
}
static void bw_put_bit(BitWriter* bw, uint8_t b){
    uint32_t byte = bw->bitpos >> 3;
    uint8_t  off  = bw->bitpos & 7;
    if (byte >= bw->capBytes) return; // simples proteção
    if (off==0) bw->buf[byte]=0;
    if (b&1) bw->buf[byte] |= (1u << (7-off));
    bw->bitpos++;
}
static void bw_put_byte(BitWriter* bw, uint8_t v){
    for (int i=7;i>=0;--i) bw_put_bit(bw, (uint8_t)((v>>i)&1));
}
static uint32_t bw_size_bytes(const BitWriter* bw){
    return (bw->bitpos + 7u) >> 3;
}

static void br_init(BitReader* br, const uint8_t* data, uint32_t sizeBytes){
    br->buf = data; br->sizeBytes = sizeBytes; br->bitpos = 0;
}
static uint8_t br_get_bit(BitReader* br){
    uint32_t byte = br->bitpos >> 3;
    uint8_t  off  = br->bitpos & 7;
    if (byte >= br->sizeBytes) return 0;
    uint8_t v = (uint8_t)((br->buf[byte] >> (7-off)) & 1u);
    br->bitpos++;
    return v;
}
static uint8_t br_get_byte(BitReader* br){
    uint8_t v=0; for (int i=7;i>=0;--i){ v |= (uint8_t)(br_get_bit(br)<<i); }
    return v;
}

// ======================= PGM P2 (ASCII) =======================
static int pgm_skip_ws_and_comments(FILE* f){
    int c;
    do {
        c = fgetc(f);
        if (c=='#'){ while (c!='\n' && c!=EOF) c = fgetc(f); }
    } while (isspace(c));
    if (c!=EOF) ungetc(c,f);
    return 1;
}

static bool pgm_read_p2(const char* path, uint8_t* img, uint16_t* W, uint16_t* H){
    FILE* f = fopen(path,"r"); if(!f){ fprintf(stderr,"erro: não abriu %s\n", path); return false; }
    char magic[3]={0}; if (fscanf(f,"%2s",magic)!=1 || magic[0]!='P' || magic[1]!='2'){ fclose(f); return false; }
    pgm_skip_ws_and_comments(f);
    int w,h,maxv;
    if (fscanf(f,"%d %d",&w,&h)!=2){ fclose(f); return false; }
    if (w<=0 || h<=0 || w>MAX_W || h>MAX_H){ fclose(f); fprintf(stderr,"dimensões inválidas (max %dx%d)\n", MAX_W, MAX_H); return false; }
    pgm_skip_ws_and_comments(f);
    if (fscanf(f,"%d",&maxv)!=1 || maxv!=255){ fclose(f); return false; }
    for (int i=0;i<w*h;i++){
        int v; if (fscanf(f,"%d",&v)!=1 || v<0 || v>255){ fclose(f); return false; }
        img[i] = (uint8_t)v;
    }
    *W=(uint16_t)w; *H=(uint16_t)h; fclose(f); return true;
}

static bool pgm_write_p2(const char* path, const uint8_t* img, uint16_t W, uint16_t H){
    FILE* f = fopen(path,"w"); if(!f) return false;
    fprintf(f,"P2\n%d %d\n255\n", W, H);
    for (uint32_t i=0;i<(uint32_t)W*H;i++){
        fprintf(f,"%d%c", img[i], ( ((i+1)%W) ? ' ' : '\n') );
    }
    fclose(f); return true;
}

// ======================= Métricas (PSNR) =======================
static double psnr_u8(const uint8_t* a, const uint8_t* b, int n){
    double s=0.0; for (int i=0;i<n;i++){ double d=(double)a[i]-b[i]; s+=d*d; }
    if (s==0.0) return 99.0;
    double mse = s / (double)n;
    return 10.0*log10( (255.0*255.0) / mse );
}

// ======================= Integrais S1/S2 =======================
// typedef struct {
//     // (H+1) x (W+1), índices 1-based nas integrais
//     uint32_t S1[(MAX_H+1)*(MAX_W+1)];
//     uint32_t S2[(MAX_H+1)*(MAX_W+1)];
//     uint16_t W, H;
// } Integrals;
typedef struct {
    uint32_t S1[(MAX_H+1)*(MAX_W+1)];
    uint64_t S2[(MAX_H+1)*(MAX_W+1)];  // << era uint32_t
    uint16_t W, H;
} Integrals;

#define IDX(I,x,y)  ((y)*((I)->W+1) + (x))

static void integral_build(Integrals* I, const uint8_t* img, uint16_t W, uint16_t H){
    I->W=W; I->H=H;
    // zera bordas
    memset(I->S1, 0, sizeof(uint32_t)*(W+1)*(H+1));
    memset(I->S2, 0, sizeof(uint32_t)*(W+1)*(H+1));

    for (uint16_t y=1; y<=H; ++y){
        uint32_t row1=0;
        uint64_t row2=0;  // << era uint32_t
        for (uint16_t x=1; x<=W; ++x){
            uint8_t p = img[(y-1)*W + (x-1)];
            row1 += p;
            row2 += (uint64_t)p * (uint64_t)p;  // << cast p² para 64 bits
            I->S1[IDX(I,x,y)] = I->S1[IDX(I,x,y-1)] + row1;
            I->S2[IDX(I,x,y)] = I->S2[IDX(I,x,y-1)] + row2;  // << S2 é uint64_t
        }
    }
}

static void integral_block_stats(const Integrals* I, uint16_t x, uint16_t y, uint16_t size,
                                 uint32_t* sum1, uint64_t* sum2, uint32_t* area){
    uint16_t x1=x+1, y1=y+1, x2=x+size, y2=y+size;
    uint32_t a = (uint32_t)size*(uint32_t)size;

    uint32_t A1 = I->S1[IDX(I,x2,y2)] - I->S1[IDX(I,x1-1,y2)]
             - I->S1[IDX(I,x2,y1-1)] + I->S1[IDX(I,x1-1,y1-1)];
    uint64_t A2 = I->S2[IDX(I,x2,y2)] - I->S2[IDX(I,x1-1,y2)]
                - I->S2[IDX(I,x2,y1-1)] + I->S2[IDX(I,x1-1,y1-1)];
    *sum1 = A1; *sum2 = A2; *area = a;
}

// ======================= Quadtree (iterativa, sem recursão) =======================
typedef struct {
    uint16_t x, y;   // origem do bloco
    uint16_t size;   // lado do bloco
    uint8_t  level;  // nível na árvore
} QTNode;            // ~8 bytes

typedef struct {
    // Pilha estática (LIFO)
    QTNode stack[MAX_STACK_NODES];
    int    top; // -1 vazio

    // Parâmetros
    uint32_t tau2;         // limiar de variância
    uint16_t s_min;        // tamanho mínimo do bloco
    uint8_t  depth_max;    // profundidade máxima

    // Dimensões da imagem
    uint16_t W, H, N;      // N = min(W,H)
} QTContext;

static inline int stack_empty(QTContext* c){ return c->top < 0; }
static inline void stack_push(QTContext* c, QTNode n){ c->stack[++c->top] = n; }
static inline QTNode stack_pop(QTContext* c){ return c->stack[c->top--]; }

static void qt_init(QTContext* c, uint16_t W, uint16_t H,
                    uint32_t tau2, uint16_t s_min, uint8_t depth_max) {
    c->W = W; c->H = H; c->N = (W<H?W:H);
    c->tau2 = tau2; c->s_min = s_min; c->depth_max = depth_max;
    c->top = -1;
}

// Escreve stream em pré-ordem: bit (0 folha / 1 interno); se folha, +1 byte (média)
static void qt_encode(const Integrals* I, QTContext* c, BitWriter* bw) {
    uint16_t N = (c->W < c->H ? c->W : c->H);
    QTNode root = (QTNode){ .x=0, .y=0, .size=N, .level=0 };
    stack_push(c, root);

    while (!stack_empty(c)) {
        QTNode b = stack_pop(c);

        uint32_t s1, area;
        uint64_t s2;  // << era uint32_t

        integral_block_stats(I, b.x, b.y, b.size, &s1, &s2, &area);
        double mean = (double)s1 / (double)area;
        double var  = (double)s2 / (double)area - (mean*mean);

        if (b.level == 0) {
            printf("[DEBUG] root: mean=%.3f  var=%.3f  tau2=%u\n", mean, var, c->tau2);
        }

        int stop = (var <= c->tau2) || (b.size <= c->s_min) || (b.level >= c->depth_max);
        if (stop) {
            uint8_t m = (uint8_t)(mean + 0.5);
            bw_put_bit(bw, 0); // folha
            bw_put_byte(bw, m);
        } else {
            bw_put_bit(bw, 1); // interno
            uint16_t half = b.size >> 1;
            uint8_t  lv   = b.level + 1;
            // Empilha em ordem inversa ao consumo: SE, SW, NE, NW => saída: NW,NE,SW,SE
            stack_push(c, (QTNode){ b.x+half, b.y+half, half, lv }); // SE
            stack_push(c, (QTNode){ b.x,      b.y+half, half, lv }); // SW
            stack_push(c, (QTNode){ b.x+half, b.y,      half, lv }); // NE
            stack_push(c, (QTNode){ b.x,      b.y,      half, lv }); // NW
        }
    }
}

// Decodifica o stream e preenche blocos
static void qt_decode_fill(BitReader* br, uint8_t* out, uint16_t W, uint16_t H,
                           uint16_t N, uint16_t s_min, uint8_t depth_max) {
    QTContext c;
    qt_init(&c, W, H, 0, s_min, depth_max);
    stack_push(&c, (QTNode){0,0,N,0});

    while (!stack_empty(&c)) {
        QTNode b = stack_pop(&c);
        uint8_t flag = br_get_bit(br);
        if (flag == 0) {
            uint8_t m = br_get_byte(br);
            for (uint16_t dy=0; dy<b.size && (b.y+dy)<H; ++dy) {
                uint8_t* row = out + (b.y+dy)*W + b.x;
                for (uint16_t dx=0; dx<b.size && (b.x+dx)<W; ++dx) row[dx] = m;
            }
        } else {
            uint16_t half = b.size >> 1; uint8_t lv = b.level+1;
            stack_push(&c, (QTNode){ b.x+half, b.y+half, half, lv });
            stack_push(&c, (QTNode){ b.x,      b.y+half, half, lv });
            stack_push(&c, (QTNode){ b.x+half, b.y,      half, lv });
            stack_push(&c, (QTNode){ b.x,      b.y,      half, lv });
        }
    }
}

// ======================= CLI commands =======================
static void cmd_compress(const char* in, const char* out, uint32_t tau2, uint16_t smin, uint8_t dmax){
    uint8_t img[MAX_W*MAX_H];
    uint16_t W,H;
    if(!pgm_read_p2(in,img,&W,&H)){ fprintf(stderr,"PGM inválido\n"); return; }

    Integrals I; integral_build(&I, img, W, H);
    QTContext ctx; qt_init(&ctx, W, H, tau2, smin, dmax);

    static uint8_t stream[1<<20]; // 1MB para PC
    BitWriter bw; bw_init(&bw, stream, sizeof(stream));

    double t0=now_ms();
    qt_encode(&I, &ctx, &bw);
    double t1=now_ms();

    FILE* f=fopen(out,"wb"); if(!f){ fprintf(stderr,"erro abrindo %s\n", out); return; }
    fwrite(stream,1,bw_size_bytes(&bw),f); fclose(f);
    printf("OK: %s -> %s | stream=%u bytes | time=%.2f ms\n", in,out,bw_size_bytes(&bw), (t1-t0));
}

static void cmd_decompress(const char* in, const char* out, uint16_t W, uint16_t H, uint16_t smin, uint8_t dmax){
    FILE* f=fopen(in,"rb"); if(!f){ fprintf(stderr,"erro abrindo %s\n", in); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    static uint8_t stream[1<<20]; fread(stream,1,(size_t)n,f); fclose(f);

    static uint8_t recon[MAX_W*MAX_H];
    memset(recon,0,sizeof(recon));

    BitReader br; br_init(&br, stream, (uint32_t)n);
    uint16_t N = (W<H?W:H);
    double t0=now_ms();
    qt_decode_fill(&br, recon, W, H, N, smin, dmax);
    double t1=now_ms();

    pgm_write_p2(out, recon, W, H);
    printf("OK: %s -> %s | time=%.2f ms\n", in,out,(t1-t0));
}

static void cmd_stats(const char* inPGM, const char* streamBin, const char* outPGM){
    uint8_t orig[MAX_W*MAX_H], recon[MAX_W*MAX_H];
    uint16_t W,H; if(!pgm_read_p2(inPGM,orig,&W,&H)){ fprintf(stderr,"PGM inválido\n"); return; }

    FILE* f=fopen(streamBin,"rb"); if(!f){ fprintf(stderr,"erro abrindo %s\n", streamBin); return; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    static uint8_t stream[1<<20]; fread(stream,1,(size_t)n,f); fclose(f);
    BitReader br; br_init(&br, stream, (uint32_t)n);

    memset(recon,0,sizeof(recon));
    uint16_t N=(W<H?W:H);
    qt_decode_fill(&br, recon, W, H, N, DEFAULT_S_MIN, DEFAULT_DEPTH_MAX);

    if (outPGM) pgm_write_p2(outPGM,recon,W,H);

    double psnr = psnr_u8(orig, recon, (int)(W*H));
    printf("bytes_in=%d bytes_out=%ld ratio=%.2f%% PSNR=%.2f dB\n",
           W*H, n, 100.0*(double)n/(double)(W*H), psnr);
}

// ======================= main =======================
int main(int argc, char** argv){
    if (argc<2){
        puts("usage:\n"
             "  compress in.pgm out.qt [-t tau2] [-s smin] [-d dmax]\n"
             "  decompress in.qt out.pgm W H [-s smin] [-d dmax]\n"
             "  stats in.pgm in.qt [out_recon.pgm]");
        return 0;
    }

    if (!strcmp(argv[1],"compress")){
        if (argc < 4){ fprintf(stderr,"compress: faltam argumentos\n"); return 1; }
        const char* in=argv[2], *out=argv[3];
        uint32_t tau2=DEFAULT_TAU2; uint16_t smin=DEFAULT_S_MIN; uint8_t dmax=DEFAULT_DEPTH_MAX;
        for (int i=4;i<argc;i++){
            if (!strcmp(argv[i],"-t") && i+1<argc) tau2=(uint32_t)atoi(argv[++i]);
            else if (!strcmp(argv[i],"-s") && i+1<argc) smin=(uint16_t)atoi(argv[++i]);
            else if (!strcmp(argv[i],"-d") && i+1<argc) dmax=(uint8_t)atoi(argv[++i]);
        }
        cmd_compress(in,out,tau2,smin,dmax);
    } else if (!strcmp(argv[1],"decompress")){
        if (argc < 6){ fprintf(stderr,"decompress: faltam argumentos\n"); return 1; }
        const char* in=argv[2], *out=argv[3];
        uint16_t W=(uint16_t)atoi(argv[4]), H=(uint16_t)atoi(argv[5]);
        uint16_t smin=DEFAULT_S_MIN; uint8_t dmax=DEFAULT_DEPTH_MAX;
        for (int i=6;i<argc;i++){
            if (!strcmp(argv[i],"-s") && i+1<argc) smin=(uint16_t)atoi(argv[++i]);
            else if (!strcmp(argv[i],"-d") && i+1<argc) dmax=(uint8_t)atoi(argv[++i]);
        }
        cmd_decompress(in,out,W,H,smin,dmax);
    } else if (!strcmp(argv[1],"stats")){
        if (argc < 4){ fprintf(stderr,"stats: faltam argumentos\n"); return 1; }
        const char* inPGM=argv[2], *inQT=argv[3]; const char* outPGM=(argc>4?argv[4]:NULL);
        cmd_stats(inPGM,inQT,outPGM);
    } else {
        puts("unknown command"); return 1;
    }
    return 0;
}