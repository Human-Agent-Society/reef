/*
SOURCE: Shang Zhou
IIMOC SUBMISSION #1443
HUMAN BEST FOR POLYPACK (TRANSFORMATION OUTPUT FORMAT)
*/

#include <bits/stdc++.h>
using namespace std;
struct T{int w,h;vector<pair<int,int>> c;vector<int> lo,hi;int r,f,minx,miny;};
struct P{int id,k;vector<pair<int,int>> b;vector<T> t;int minW=1e9,minH=1e9,minA=1e9;};
struct Pl{int idx,ti,x,y;};
struct R{long long A;int W,H;vector<Pl> pl;};
struct RNG{unsigned long long s;RNG(unsigned long long x){s=x?x:1;}inline unsigned long long nxt(){s^=s<<7;s^=s>>9;return s;}inline int rint(int n){return (int)(nxt()%n);}inline bool coin(){return nxt()&1;}};
static inline pair<int,int> rotp(pair<int,int> p,int r){if(r==0)return p; if(r==1)return make_pair(-p.second,p.first); if(r==2)return make_pair(-p.first,-p.second); return make_pair(p.second,-p.first);}
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n; if(!(cin>>n)) return 0;
    vector<P> ps(n); long long S=0;
    for(int i=0;i<n;i++){
        int k;cin>>k; ps[i].id=i+1; ps[i].k=k; ps[i].b.resize(k);
        for(int j=0;j<k;j++){int x,y;cin>>x>>y; ps[i].b[j]={x,y};}
        S+=k;
    }
    for(int i=0;i<n;i++){
        auto &p=ps[i]; unordered_set<string> seen; seen.reserve(32);
        for(int rf=0;rf<2;rf++){
            vector<pair<int,int>> src=p.b; if(rf){for(auto &q:src) q.first=-q.first;}
            for(int r=0;r<4;r++){
                vector<pair<int,int>> v=src; for(auto &q:v) q=rotp(q,r);
                int minx=INT_MAX,miny=INT_MAX,maxx=INT_MIN,maxy=INT_MIN;
                for(auto &q:v){minx=min(minx,q.first);miny=min(miny,q.second);maxx=max(maxx,q.first);maxy=max(maxy,q.second);}
                vector<pair<int,int>> v2=v; for(auto &q:v2){q.first-=minx;q.second-=miny;}
                sort(v2.begin(),v2.end());
                string key; key.reserve(v2.size()*8);
                for(auto &q:v2){key.append(to_string(q.first));key.push_back(',');key.append(to_string(q.second));key.push_back(';');}
                if(seen.insert(key).second){
                    T t; t.w=maxx-minx+1; t.h=maxy-miny+1; t.c=v2; t.r=r; t.f=rf; t.minx=minx; t.miny=miny;
                    t.lo.assign(t.w,INT_MAX); t.hi.assign(t.w,INT_MIN);
                    for(auto &q:v2){int x=q.first,y=q.second; if(t.lo[x]>y) t.lo[x]=y; if(t.hi[x]<y) t.hi[x]=y;}
                    p.t.push_back(move(t));
                }
            }
        }
        for(auto &t:p.t){p.minW=min(p.minW,t.w); p.minH=min(p.minH,t.h); p.minA=min(p.minA,t.w*t.h);}
        if(p.t.empty()){T t; t.w=1;t.h=1;t.c={{0,0}};t.lo={0};t.hi={0};t.r=0;t.f=0;t.minx=0;t.miny=0;p.t.push_back(t);p.minW=1;p.minH=1;p.minA=1;}
    }
    vector<int> idx(n); iota(idx.begin(),idx.end(),0);
    unsigned long long seed=((unsigned long long)chrono::high_resolution_clock::now().time_since_epoch().count()) ^ (S<<1) ^ (unsigned long long)(n*1469598103934665603ULL);
    RNG rng(seed);
    auto ord4 = [&]() {
        vector<int> res = idx;
        stable_sort(res.begin(), res.end(), [&](int a, int b) {
            int da = min(ps[a].minW, ps[a].minH);
            int db = min(ps[b].minW, ps[b].minH);
            if (da != db) return da < db;
            if (ps[a].k != ps[b].k) return ps[a].k < ps[b].k;
            return ps[a].id > ps[b].id;
        });
        return res;
    };
    auto pack=[&](int W,const vector<int>& o0,RNG &rng,bool randtie){
        vector<int> h(W,-1);
        long long g=-1;
        vector<Pl> pl; pl.reserve(o0.size());
        vector<int> o=o0;
        int t=0,nm=(int)o.size();
        bool big=S>7000;
        int maxBound=max(1,n/4);
        int expLIM=min(maxBound,(int)(350000/max(1LL,S-3500)));
        int dynLIM=big?expLIM:maxBound;
        auto tStart=chrono::steady_clock::now();
        auto batchStart=tStart;
        long long TLms=big?1900:LLONG_MAX/4;
        int stepCnt=0;
        while(t<nm){
            int limCnt=max(1,dynLIM);
            int lim=min(nm,t+limCnt);
            long long bestg=LLONG_MAX; int bti=-1,bx=0,by=0,bl=INT_MAX; long long bds=LLONG_MAX; long long bdr=LLONG_MAX; int by0=INT_MAX,bx0=INT_MAX; int bestpos=t; int bestid=-1;
            for(int pos=t;pos<lim;pos++){
                int id=o[pos];
                auto &p=ps[id];
                long long bestg2=LLONG_MAX; int bti2=-1,bx2=0,by2=0,bl2=INT_MAX; long long bds2=LLONG_MAX; long long bdr2=LLONG_MAX; int by02=INT_MAX,bx02=INT_MAX;
                for(int ti=0;ti<(int)p.t.size();ti++){
                    auto &tsh=p.t[ti]; if(tsh.w>W) continue; int Rpos=W-tsh.w+1;
                    for(int x0=0;x0<Rpos;x0++){
                        int y0=0;
                        for(int j=0;j<tsh.w;j++){
                            if(tsh.lo[j]!=INT_MAX){int v=h[x0+j]-tsh.lo[j]+1; if(v>y0) y0=v;}
                        }
                        int nhbuf[32];
                        int l=-1; long long dsum=0;
                        for(int j=0;j<tsh.w;j++){
                            int nh=h[x0+j];
                            if(tsh.hi[j]!=INT_MIN){
                                int cand=y0+tsh.hi[j];
                                if(nh<cand) nh=cand;
                            }
                            nhbuf[j]=nh;
                            if(nh>l) l=nh;
                        }
                        for(int j=0;j<tsh.w;j++){
                            int inc=nhbuf[j]-h[x0+j];
                            if(inc>0) dsum+=inc;
                        }
                        long long dr=0;
                        if(x0>0){
                            long long old=llabs((long long)h[x0]-h[x0-1]);
                            long long nw=llabs((long long)nhbuf[0]-h[x0-1]);
                            dr+=nw-old;
                        }
                        for(int j=0;j<tsh.w-1;j++){
                            long long old=llabs((long long)h[x0+j+1]-h[x0+j]);
                            long long nw=llabs((long long)nhbuf[j+1]-nhbuf[j]);
                            dr+=nw-old;
                        }
                        if(x0+tsh.w<W){
                            long long old=llabs((long long)h[x0+tsh.w]-h[x0+tsh.w-1]);
                            long long nw=llabs((long long)h[x0+tsh.w]-nhbuf[tsh.w-1]);
                            dr+=nw-old;
                        }
                        long long gg=g; if(gg<l) gg=l;
                        bool take=false;
                        if(gg<bestg2) take=true;
                        else if(gg==bestg2){
                            if(dsum<bds2) take=true;
                            else if(dsum==bds2){
                                if(l<bl2) take=true;
                                else if(l==bl2){
                                    if(dr<bdr2) take=true;
                                    else if(dr==bdr2){
                                        if(y0<by02) take=true;
                                        else if(y0==by02){
                                            if(x0<bx02) take=true;
                                            else if(x0==bx02 && randtie && rng.coin()) take=true;
                                        }
                                    }
                                }
                            }
                        }
                        if(take){bestg2=gg;bti2=ti;bx2=x0;by2=y0;bl2=l;bds2=dsum;bdr2=dr;by02=y0;bx02=x0;}
                    }
                }
                if(bti2==-1) continue;
                bool take=false;
                if(bestg2<bestg) take=true;
                else if(bestg2==bestg){
                    if(bds2<bds) take=true;
                    else if(bds2==bds){
                        if(bl2<bl) take=true;
                        else if(bl2==bl){
                            if(bdr2<bdr) take=true;
                            else if(bdr2==bdr){
                                if(by02<by0) take=true;
                                else if(by02==by0){
                                    if(bx02<bx0) take=true;
                                    else if(bx02==bx0 && randtie && rng.coin()) take=true;
                                }
                            }
                        }
                    }
                }
                if(take){bestg=bestg2;bti=bti2;bx=bx2;by=by2;bl=bl2;bds=bds2;bdr=bdr2;by0=by02;bx0=bx02;bestpos=pos;bestid=id;}
            }
            if(bti==-1){t++;stepCnt++;if(big&&stepCnt==5){auto now=chrono::steady_clock::now(); auto elapsed=chrono::duration<double,milli>(now-tStart).count(); auto batch=chrono::duration<double,milli>(now-batchStart).count(); double remT=max(0.0,TLms-elapsed); int remSteps=max(1,nm-t); double budget=remT*5.0/remSteps; if(batch<budget) dynLIM=min(maxBound,dynLIM+1); else dynLIM=max(1,dynLIM-1); batchStart=now; stepCnt=0;} continue;}
            auto &tsh=ps[bestid].t[bti];
            for(int j=0;j<tsh.w;j++){
                int nh=h[bx+j];
                if(tsh.hi[j]!=INT_MIN){
                    int cand=by+tsh.hi[j];
                    if(nh<cand) nh=cand;
                }
                h[bx+j]=nh;
            }
            if(g<bl) g=bl;
            pl.push_back({bestid,bti,bx,by});
            if(bestpos!=t) swap(o[t],o[bestpos]);
            t++;
            stepCnt++;
            if(big&&stepCnt==5){
                auto now=chrono::steady_clock::now();
                auto elapsed=chrono::duration<double,milli>(now-tStart).count();
                auto batch=chrono::duration<double,milli>(now-batchStart).count();
                double remT=max(0.0,TLms-elapsed);
                int remSteps=max(1,nm-t);
                double budget=remT*5.0/remSteps;
                if(batch<budget) dynLIM=min(maxBound,dynLIM+1); else dynLIM=max(1,dynLIM-1);
                batchStart=now;
                stepCnt=0;
            }
        }
        int H=(int)g+1;
        int maxX=-1;
        for(auto &pp:pl){
            auto &t=ps[pp.idx].t[pp.ti];
            for(auto &q:t.c){int x=pp.x+q.first; if(x>maxX) maxX=x;}
        }
        int Wused=0;
        if(maxX>=0){
            vector<char> used(maxX+1,false);
            for(auto &pp:pl){auto &t=ps[pp.idx].t[pp.ti]; for(auto &q:t.c){used[pp.x+q.first]=true;}}
            for(int x=0;x<=maxX;x++) if(used[x]) Wused++;
        }
        int Wfinal=max(0,Wused);
        long long A=1LL*H*max(1,Wfinal);
        return R{A,max(1,Wfinal),H,move(pl)};
    };
    int minW=0; for(auto &p:ps) minW=max(minW,p.minW);
    double factor;
    if (S < 1000) {
        factor = 0.4;
    } else if (S < 3000) {
        factor = 0.5;
    } else if (S < 10000) {
        factor = 0.27;
    } else if (S < 30000) {
        factor = 0.08;
    } else {
        factor = 0.01;
    }
    int base = max(minW, (int)floor(sqrt((double)S * factor)));
    vector<int> Ws;
    {
        unordered_set<int> used; used.reserve(512);
        auto addW=[&](int w){if(w<minW) w=minW; if(used.insert(w).second) Ws.push_back(w);};
        addW(base);
        int span=min(96,max(20,base/2));
        for(int d=1;d<=span;d++){addW(base-d); addW(base+d);}
        addW(minW);
        addW((int)max<long long>(minW,(S+base-1)/base));
        for(int m=2;m<=6;m++){addW(base*m/3); addW((int)max<long long>(minW,S/((base*m/3)?(base*m/3):1)));}
        sort(Ws.begin(),Ws.end(),[&](int a,int b){int da=abs(a-base),db=abs(b-base); if(da!=db) return da<db; return a<b;});
    }
    long long bestA=LLONG_MAX; int bestW=0,bestH=0; R bestR; bool hasBestmine=false;
    auto t0=chrono::steady_clock::now();
    const double TL=1980.0;
    double avg=250.0; int cnt=0;
    for(int wi=0;wi<(int)Ws.size();wi++){
        auto now=chrono::steady_clock::now();
        double used=chrono::duration<double,milli>(now-t0).count();
        if(used+avg*1.3>TL) break;
        int W=Ws[wi];
        vector<vector<int>> orders;
        orders.push_back(ord4());
        int oi=0;
        while(oi<(int)orders.size()){
            auto t1=chrono::steady_clock::now();
            double used2=chrono::duration<double,milli>(t1-t0).count();
            if(used2+avg*1.15>TL) break;
            bool randtie=(oi>=1);
            R r=pack(W,orders[oi],rng,randtie);
            auto t2=chrono::steady_clock::now();
            double dt=chrono::duration<double,milli>(t2-t1).count();
            cnt++; avg=(avg*(cnt-1)+dt)/cnt;
            if(!hasBestmine || r.A<bestA || (r.A==bestA && (r.H<bestH || (r.H==bestH && r.W<bestW)))){
                bestA=r.A; bestW=r.W; bestH=r.H; bestR=r; hasBestmine=true;
            }
            oi++;
        }
    }
    if(!hasBestmine){
        int W=max(minW,(int)floor(sqrt((double)S)));
        auto o=ord4();
        R r=pack(W,o,rng,false);
        bestA=r.A; bestW=r.W; bestH=r.H; bestR=r; hasBestmine=true;
    }
    int maxX=-1;
    for(auto &p:bestR.pl){
        auto &t=ps[p.idx].t[p.ti];
        for(auto &q:t.c){int x=p.x+q.first; if(x>maxX) maxX=x;}
    }
    vector<int> mapx(maxX+1,-1);
    if(maxX>=0){
        vector<char> used(maxX+1,false);
        for(auto &p:bestR.pl){
            auto &t=ps[p.idx].t[p.ti];
            for(auto &q:t.c){used[p.x+q.first]=true;}
        }
        int cur=0;
        for(int x=0;x<=maxX;x++) if(used[x]) mapx[x]=cur++;
    }
    vector<array<int,4>> ans(n,{0,0,0,0});
    for(auto &p:bestR.pl){
        auto &t=ps[p.idx].t[p.ti];
        int bx = (mapx.empty()?p.x:mapx[p.x]);
        int Xi = bx - t.minx;
        int Yi = p.y - t.miny;
        int Ri = (4 - (t.r%4) + 4) % 4;
        int Fi = t.f;
        ans[p.idx]={Xi,Yi,Ri,Fi};
    }
    cout<<bestR.W<<" "<<bestR.H<<"\n";
    for(int i=0;i<n;i++){
        cout<<ans[i][0]<<" "<<ans[i][1]<<" "<<ans[i][2]<<" "<<ans[i][3]<<"\n";
    }
    return 0;
}