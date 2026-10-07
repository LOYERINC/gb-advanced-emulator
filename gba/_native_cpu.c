#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <math.h>

#define MASK32 0xffffffffu
#define N 0x80000000u
#define Z 0x40000000u
#define C 0x20000000u
#define V 0x10000000u
#define T 0x20u

static uint32_t sx(uint32_t x, unsigned bits) {
    uint32_t m = 1u << (bits - 1);
    return (x ^ m) - m;
}
static void nz(uint32_t *ps, uint32_t x) {
    *ps = (*ps & ~(N|Z)) | (x & N) | (x == 0 ? Z : 0);
}
static void addflags(uint32_t *ps, uint32_t a, uint32_t b, uint32_t r) {
    *ps = (*ps & ~(N|Z|C|V)) | (r & N) | (r == 0 ? Z : 0)
        | ((uint64_t)a + b > MASK32 ? C : 0)
        | ((~(a ^ b) & (a ^ r) & N) ? V : 0);
}
static void subflags(uint32_t *ps, uint32_t a, uint32_t b, uint32_t r) {
    *ps = (*ps & ~(N|Z|C|V)) | (r & N) | (r == 0 ? Z : 0)
        | (a >= b ? C : 0) | (((a ^ b) & (a ^ r) & N) ? V : 0);
}
static int condition(uint32_t ps, unsigned c) {
    int n=!!(ps&N), z=!!(ps&Z), carry=!!(ps&C), v=!!(ps&V);
    switch(c) { case 0:return z; case 1:return !z; case 2:return carry; case 3:return !carry;
    case 4:return n; case 5:return !n; case 6:return v; case 7:return !v;
    case 8:return carry&&!z; case 9:return !carry||z; case 10:return n==v;
    case 11:return n!=v; case 12:return !z&&(n==v); case 13:return z||(n!=v);
    case 14:return 1; default:return 0; }
}
static int fetch_extra(uint32_t addr, uint32_t prev, int sequential, uint16_t waitcnt) {
    if (addr < 0x08000000u || addr >= 0x0e000000u) return 0;
    unsigned bank=(addr>>25)-4;
    static const int first[4]={4,3,2,8};
    int shift=bank==0?2:(bank==1?5:8);
    int nonseq=first[(waitcnt>>shift)&3];
    int seq;
    if(bank==0) seq=(waitcnt&0x10)?1:2;
    else if(bank==1) seq=(waitcnt&0x80)?1:4;
    else seq=(waitcnt&0x400)?1:8;
    if((addr&0x1ffffu)==0) sequential=0;
    if(sequential && prev+2==addr && ((prev>>25)-4)==bank) return seq;
    return nonseq;
}
static int ram_region(uint32_t addr, unsigned width, Py_ssize_t *size,
                      uint32_t *offset, int *extra) {
    uint32_t base=addr&0xff000000u;
    if(base==0x02000000u){*size=0x40000;*offset=(addr-0x02000000u)&0x3ffff;*extra=(width==4)?5:2;}
    else if(base==0x03000000u){*size=0x8000;*offset=(addr-0x03000000u)&0x7fff;*extra=0;}
    else return 0;
    return 1;
}
static int ram_access(uint32_t addr, unsigned width, unsigned char *ew, unsigned char *iw,
                      unsigned char **ptr, Py_ssize_t *size, uint32_t *off, int *extra) {
    if(!ram_region(addr,width,size,off,extra))return 0;
    *ptr=((addr&0xff000000u)==0x02000000u)?ew:iw;return 1;
}
static uint32_t ram_read(const unsigned char *p, Py_ssize_t size, uint32_t off, unsigned width) {
    uint32_t v=0; for(unsigned i=0;i<width;i++) v|=(uint32_t)p[(off+i)&(uint32_t)(size-1)]<<(8*i); return v;
}
static uint32_t rom_offset(uint64_t offset, Py_ssize_t size) {
    uint64_t length=(uint64_t)size;
    if((length & (length-1))==0) return (uint32_t)(offset & (length-1));
    return (uint32_t)(offset % length);
}
static void ram_write(unsigned char *p, Py_ssize_t size, uint32_t off, unsigned width, uint32_t v) {
    for(unsigned i=0;i<width;i++) p[(off+i)&(uint32_t)(size-1)]=(unsigned char)(v>>(8*i));
}
static uint32_t rom_read(const unsigned char *p, Py_ssize_t size, uint32_t off, unsigned width) {
    uint32_t v=0;for(unsigned i=0;i<width;i++)v|=(uint32_t)p[rom_offset((uint64_t)off+i,size)]<<(8*i);return v;
}
static int bus_transfer(PyObject *bus, const unsigned char *ioreg, int load, unsigned width, uint32_t addr,
                        uint32_t value, uint32_t *read_value, uint32_t *extra) {
    /* Display control/status registers are plain readable state updated by
       advance_cycles(); read these directly while retaining Python bus calls
       for timers, keypad, audio, save hardware, and all writes. */
    if(load && addr>=0x04000000u && addr+width-1<=0x04000055u){
        unsigned offset=addr-0x04000000u;*read_value=0;
        for(unsigned i=0;i<width;i++)*read_value|=(uint32_t)ioreg[offset+i]<<(8*i);
        *extra=0;return 0;
    }
    PyObject *obj=PyObject_CallMethod(bus,"begin_instruction_access_tracking",NULL);
    if(!obj)return -1;Py_DECREF(obj);
    if(load){
        const char *name=width==1?"read8":width==2?"read16":"read32";
        obj=PyObject_CallMethod(bus,name,"I",addr);
        if(!obj){PyObject *cancel=PyObject_CallMethod(bus,"cancel_instruction_access_tracking",NULL);Py_XDECREF(cancel);return -1;}
        *read_value=(uint32_t)PyLong_AsUnsignedLongMask(obj);Py_DECREF(obj);
        if(PyErr_Occurred()){PyObject *cancel=PyObject_CallMethod(bus,"cancel_instruction_access_tracking",NULL);Py_XDECREF(cancel);return -1;}
    } else {
        const char *name=width==1?"write8":width==2?"write16":"write32";
        obj=PyObject_CallMethod(bus,name,"II",addr,value);
        if(!obj){PyObject *cancel=PyObject_CallMethod(bus,"cancel_instruction_access_tracking",NULL);Py_XDECREF(cancel);return -1;}
        Py_DECREF(obj);
    }
    obj=PyObject_CallMethod(bus,"finish_instruction_access_tracking",NULL);
    if(!obj)return -1;*extra=(uint32_t)PyLong_AsUnsignedLong(obj);Py_DECREF(obj);
    if(PyErr_Occurred())return -1;
    return 0;
}

/* Run compute-only Thumb opcodes in batches. Any instruction requiring a bus,
   BIOS, mode switch or unimplemented behavior is left for Arm7Tdmi.step(). */
static PyObject *run_thumb_batch(PyObject *self, PyObject *args) {
    PyObject *cpu, *regs_obj, *bus, *cart, *rom_obj, *io_obj;
    int limit, quantum=256, initial_pending=0;
    if(!PyArg_ParseTuple(args,"Oi|ii",&cpu,&limit,&quantum,&initial_pending)) return NULL;
    if(limit<=0) return Py_BuildValue("(iii)",0,initial_pending,0);
    if(quantum<=0) quantum=256;
    PyObject *thumb=PyObject_GetAttrString(cpu,"cpsr"); if(!thumb) return NULL;
    uint32_t ps=(uint32_t)PyLong_AsUnsignedLongMask(thumb); Py_DECREF(thumb);
    if(!(ps&T)) return Py_BuildValue("(iii)",0,0,0);
    PyObject *flag=PyObject_GetAttrString(cpu,"_halted");if(!flag)return NULL;int sleeping=PyObject_IsTrue(flag);Py_DECREF(flag);
    if(sleeping)return Py_BuildValue("(iii)",0,0,0);
    flag=PyObject_GetAttrString(cpu,"_stopped");if(!flag)return NULL;sleeping=PyObject_IsTrue(flag);Py_DECREF(flag);
    if(sleeping)return Py_BuildValue("(iii)",0,0,0);
    flag=PyObject_GetAttrString(cpu,"_intr_wait_mask");if(!flag)return NULL;unsigned long wait_mask=PyLong_AsUnsignedLong(flag);Py_DECREF(flag);
    if(wait_mask)return Py_BuildValue("(iii)",0,0,0);
    bus=PyObject_GetAttrString(cpu,"bus"); if(!bus) return NULL;
    flag=PyObject_GetAttrString(bus,"_low_power_request");if(!flag){Py_DECREF(bus);return NULL;}
    int low_power=(flag!=Py_None);Py_DECREF(flag);if(low_power){Py_DECREF(bus);return Py_BuildValue("(iii)",0,0,0);}
    PyObject *bios=PyObject_GetAttrString(bus,"bios_data");
    if(!bios){Py_DECREF(bus);return NULL;}
    int hasbios=(bios!=Py_None); Py_DECREF(bios);
    if(hasbios){Py_DECREF(bus);return Py_BuildValue("(iii)",0,0,0);}
    cart=PyObject_GetAttrString(bus,"cartridge"); if(!cart){Py_DECREF(bus);return NULL;}
    rom_obj=PyObject_GetAttrString(cart,"data"); Py_DECREF(cart);
    if(!rom_obj){Py_DECREF(bus);return NULL;}
    Py_buffer rom;
    if(PyObject_GetBuffer(rom_obj,&rom,PyBUF_SIMPLE)<0){Py_DECREF(rom_obj);Py_DECREF(bus);return NULL;}
    PyObject *io=PyObject_GetAttrString(bus,"io");
    PyObject *ew=PyObject_GetAttrString(bus,"ewram");
    PyObject *iw=PyObject_GetAttrString(bus,"iwram");
    if(!io||!ew||!iw){Py_XDECREF(io);Py_XDECREF(ew);Py_XDECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;}
    Py_buffer iov;
    Py_buffer ewv,iwv;
    if(PyObject_GetBuffer(io,&iov,PyBUF_SIMPLE)<0){Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;}
    if(PyObject_GetBuffer(ew,&ewv,PyBUF_SIMPLE)<0){PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;}
    if(PyObject_GetBuffer(iw,&iwv,PyBUF_SIMPLE)<0){PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;}
    regs_obj=PyObject_GetAttrString(cpu,"registers");
    if(!regs_obj){PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;}
    if(!PyList_Check(regs_obj)||PyList_GET_SIZE(regs_obj)<16||rom.len<2){
        Py_DECREF(regs_obj);PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return Py_BuildValue("(iii)",0,0,0);
    }
    uint32_t r[16];
    for(int i=0;i<16;i++){r[i]=(uint32_t)PyLong_AsUnsignedLongMask(PyList_GET_ITEM(regs_obj,i)); if(PyErr_Occurred()) goto fail;}
    uint32_t prev=0; int prev_valid=0; int last_valid=0; uint32_t last_addr=0;
    PyObject *last=PyObject_GetAttrString(cpu,"_last_gamepak_fetch");
    if(last && last!=Py_None && PyTuple_Check(last) && PyTuple_GET_SIZE(last)>=3){
        last_addr=(uint32_t)PyLong_AsUnsignedLongMask(PyTuple_GET_ITEM(last,0));
        int w=(int)PyLong_AsLong(PyTuple_GET_ITEM(last,1)); int t=PyObject_IsTrue(PyTuple_GET_ITEM(last,2));
        last_valid=(w==2 && t==1); prev=last_addr; prev_valid=last_valid;
    }
    Py_XDECREF(last); if(PyErr_Occurred()) goto fail;
    uint16_t waitcnt=((const unsigned char*)iov.buf)[0x204] | ((((const unsigned char*)iov.buf)[0x205]&0x7f)<<8);
    unsigned char *ioreg=(unsigned char*)iov.buf;
    if(!(ps&0x80u) && (ioreg[0x208]&1) &&
       ((ioreg[0x200]|((uint16_t)ioreg[0x201]<<8)) & (ioreg[0x202]|((uint16_t)ioreg[0x203]<<8)))){
        Py_DECREF(regs_obj);PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return Py_BuildValue("(iii)",0,0,1);
    }
    const unsigned char *code=(const unsigned char*)rom.buf; Py_ssize_t romlen=rom.len;
    uint32_t total_cycles=0,pending_cycles=(uint32_t)initial_pending; int done=0, stop=0; int data_valid=0; uint32_t data_addr=0; unsigned data_width=0;
    while(done<limit){
        uint32_t cycles_before=total_cycles;
        uint32_t addr=r[15];
        if(addr<0x08000000u || addr>=0x0e000000u) {stop=1;break;}
        Py_ssize_t off=(Py_ssize_t)rom_offset(addr-0x08000000u,romlen);
        if(off+1>=romlen){stop=1;break;}
        uint16_t op=(uint16_t)(code[off]|((uint16_t)code[off+1]<<8));
        uint32_t next=(addr+2)&MASK32; int base=2; int supported=1;
        if((op&0xe000)==0x2000){
            unsigned kind=(op>>11)&3, d=(op>>8)&7, imm=op&255; uint32_t old=r[d],res;
            if(kind==0){r[d]=imm;nz(&ps,imm);} else if(kind==1){res=old-imm;subflags(&ps,old,imm,res);}
            else if(kind==2){res=old+imm;r[d]=res;addflags(&ps,old,imm,res);} else {res=old-imm;r[d]=res;subflags(&ps,old,imm,res);}
        } else if((op&0xf800)==0x0000 || (op&0xf800)==0x0800 || (op&0xf800)==0x1000){
            unsigned typ=(op>>11)&3, amount=(op>>6)&31, s=(op>>3)&7,d=op&7; uint32_t x=r[s],res; int carry=!!(ps&C);
            if(typ==0){if(amount){carry=!!(x&(1u<<(32-amount)));res=x<<amount;}else res=x;}
            else if(typ==1){amount=amount?amount:32;carry=!!(x&(1u<<(amount-1)));res=amount==32?0:x>>amount;}
            else {amount=amount?amount:32;carry=!!(x&(1u<<(amount-1)));res=(uint32_t)(((int32_t)x)>>amount);}
            r[d]=res;nz(&ps,res);ps=(ps&~C)|(carry?C:0);
        } else if((op&0xf800)==0x1800){
            unsigned imm=(op>>10)&1, sub=(op>>9)&1, field=(op>>6)&7,s=(op>>3)&7,d=op&7;
            uint32_t a=r[s],b=imm?field:r[field],res=sub?a-b:a+b; r[d]=res;
            if(sub)subflags(&ps,a,b,res);else addflags(&ps,a,b,res);
        } else if((op&0xfc00)==0x4000){
            unsigned kind=(op>>6)&15, s=(op>>3)&7,d=op&7; uint32_t a=r[d],b=r[s],res=0; int write=1;
            if(kind==0)res=a&b;
            else if(kind==1)res=a^b;
            else if(kind==2||kind==3||kind==4||kind==7){
                unsigned amount=b&255; int carry=!!(ps&C);
                if(!amount)res=a;
                else if(kind==2){if(amount<32){carry=!!(a&(1u<<(32-amount)));res=a<<amount;}else if(amount==32){carry=!!(a&1);res=0;}else{carry=0;res=0;}}
                else if(kind==3){if(amount<32){carry=!!(a&(1u<<(amount-1)));res=a>>amount;}else if(amount==32){carry=!!(a&N);res=0;}else{carry=0;res=0;}}
                else if(kind==4){int sign=!!(a&N);if(amount>=32){carry=sign;res=sign?MASK32:0;}else{carry=!!(a&(1u<<(amount-1)));res=(uint32_t)(((int32_t)a)>>amount);}}
                else {unsigned rot=amount&31;if(!rot){res=a;carry=!!(a&N);}else{res=(a>>rot)|(a<<(32-rot));carry=!!(res&N);}}
                if(amount)ps=(ps&~C)|(carry?C:0);
            } else if(kind==5){unsigned cin=!!(ps&C);uint64_t wide=(uint64_t)a+b+cin;res=(uint32_t)wide;ps=(ps&~(N|Z|C|V))|(res&N)|(res==0?Z:0)|(wide>MASK32?C:0)|((~(a^b)&(a^res)&N)?V:0);}
            else if(kind==6){unsigned borrow=!(ps&C);uint64_t rhs=(uint64_t)b+borrow;res=(uint32_t)(a-rhs);ps=(ps&~(N|Z|C|V))|(res&N)|(res==0?Z:0)|((uint64_t)a>=rhs?C:0)|(((a^b)&(a^res)&N)?V:0);}
            else if(kind==8){res=a&b;write=0;}
            else if(kind==9){res=0-b;subflags(&ps,0,b,res);}
            else if(kind==10){res=a-b;subflags(&ps,a,b,res);write=0;}
            else if(kind==11){res=a+b;addflags(&ps,a,b,res);write=0;}
            else if(kind==12)res=a|b;
            else if(kind==13){res=a*b;unsigned v=b;if((v>>8)==0||(v>>8)==0xffffff)base=2;else if((v>>16)==0||(v>>16)==0xffff)base=3;else if((v>>24)==0||(v>>24)==0xff)base=4;else base=5;}
            else if(kind==14)res=a&~b;
            else res=~b;
            if(kind==0||kind==1||kind==2||kind==3||kind==4||kind==7||kind==8||kind==12||kind==13||kind==14||kind==15)nz(&ps,res);
            if(write)r[d]=res&MASK32;
        } else if((op&0xfc00)==0x4400){
            unsigned kind=(op>>8)&3,s=(op>>3)&15,d=(op&7)|((op>>4)&8); uint32_t right=s==15?((addr+4)&~3u):r[s];
            if(kind==3){ps=(ps|T);if(!(right&1))ps&=~T;next=right&((ps&T)?~1u:~3u);base=3;}
            else {uint32_t left=d==15?((addr+4)&~3u):r[d],res;if(kind==0){res=left+right;if(d==15)next=res&~1u;else r[d]=res;}else if(kind==1){res=left-right;subflags(&ps,left,right,res);}else if(d==15)next=right&~1u;else r[d]=right;}
        } else if((op&0xf000)==0xd000 && ((op>>8)&15)<14){
            unsigned cond=(op>>8)&15; base=condition(ps,cond)?3:1;
            if(base==3) next=(addr+4+(uint32_t)sx(op&255,8)*2)&MASK32;
        } else if((op&0xf800)==0xe000){
            base=3; next=(addr+4+(uint32_t)sx(op&0x7ff,11)*2)&MASK32;
        } else if((op&0xf800)==0xf000){
            r[14]=(addr+4+(uint32_t)sx(op&0x7ff,11)*4096u)&MASK32;base=1;
        } else if((op&0xf800)==0xf800){
            next=(r[14]+((op&0x7ff)<<1))&~1u;r[14]=((addr+2)|1u)&MASK32;base=3;
        } else if((op&0xf000)==0xa000){
            unsigned d=(op>>8)&7;uint32_t base_addr=(op&0x0800)?r[13]:((addr+4)&~3u);r[d]=base_addr+((op&255)<<2);
        } else if((op&0xff00)==0xb000){
            uint32_t amount=(op&0x7f)<<2;if(op&0x80)r[13]-=amount;else r[13]+=amount;
        } else if((op&0xf800)==0x4800){
            uint32_t location=((addr+4)&~3u)+((op&255)<<2);
            if(location<0x08000000u||location>=0x0e000000u){supported=0;}
            else {uint32_t offset=location-0x08000000u;r[(op>>8)&7]=rom_read(code,romlen,offset,4);base=3;}
        } else if((op&0xf000)==0x5000 || (op&0xe000)==0x6000 ||
                  (op&0xf000)==0x8000 || (op&0xf000)==0x9000 ||
                  (op&0xfe00)==0xb400 || (op&0xfe00)==0xbc00 || (op&0xf000)==0xc000){
            if((op&0xfe00)==0xb400 || (op&0xfe00)==0xbc00 || (op&0xf000)==0xc000){
                int stack=(op&0xfe00)==0xb400 || (op&0xfe00)==0xbc00;
                int pop=stack?!!(op&0x0800):!!(op&0x0800);
                int extra=stack?!!(op&0x0100):0;
                unsigned rb=(op>>8)&7, mask=op&255, count=0;
                for(unsigned q=0;q<8;q++)if(mask&(1u<<q))count++;
                if(stack)count+=extra; else if(count==0)count=1;
                uint32_t start;
                if(stack)start=pop?r[13]:r[13]-count*4;
                else start=r[rb];
                unsigned char *ptrs[9];Py_ssize_t sizes[9];uint32_t offs[9];int waits[9];
                for(unsigned q=0;q<count;q++){
                    uint32_t loc=stack?start+q*4:start+q*4;
                    if(!ram_access(loc,4,(unsigned char*)ewv.buf,(unsigned char*)iwv.buf,&ptrs[q],&sizes[q],&offs[q],&waits[q])){supported=0;break;}
                    offs[q]&=~3u;
                }
                if(supported){
                    for(unsigned q=0;q<count;q++)total_cycles+=(uint32_t)waits[q];
                    if(stack){
                        if(pop){unsigned q=0;for(unsigned reg=0;reg<8;reg++)if(mask&(1u<<reg)){r[reg]=ram_read(ptrs[q],sizes[q],offs[q],4);q++;}
                            if(extra)next=ram_read(ptrs[q],sizes[q],offs[q],4)&~1u;
                            r[13]=start+count*4;base=(count+2<2)?2:(int)count+2;
                        } else {unsigned q=0;for(unsigned reg=0;reg<8;reg++)if(mask&(1u<<reg)){ram_write(ptrs[q],sizes[q],offs[q],4,r[reg]);q++;}
                            if(extra)ram_write(ptrs[q],sizes[q],offs[q],4,r[14]);r[13]=start;base=(int)count+1;if(base<2)base=2;
                        }
                    } else {
                        unsigned original=mask,count_regs=0;for(unsigned reg=0;reg<8;reg++)if(mask&(1u<<reg))count_regs++;
                        uint32_t updated=start+(count_regs?count_regs*4:0x40);unsigned q=0;
                        if(!count_regs){if(pop)next=ram_read(ptrs[0],sizes[0],offs[0],4)&~1u;else ram_write(ptrs[0],sizes[0],offs[0],4,(addr+4)&~3u);}
                        else {for(unsigned reg=0;reg<8;reg++)if(mask&(1u<<reg)){if(pop)r[reg]=ram_read(ptrs[q],sizes[q],offs[q],4);else {uint32_t value=r[reg];unsigned first=0;while(first<8&&!(original&(1u<<first)))first++;if(reg==rb&&first!=rb)value=updated;ram_write(ptrs[q],sizes[q],offs[q],4,value);}q++;}}
                        if(!(pop&&(mask&(1u<<rb))))r[rb]=updated;
                        base=(int)(count_regs?count_regs:1)+(pop?2:1);
                    }
                }
            } else {
            unsigned width=4; int load=0, sign=0; uint32_t location=0, value=0;
            if((op&0xf000)==0x5000){
                unsigned kind=(op>>9)&7, ro=(op>>6)&7, rb=(op>>3)&7, rd=op&7;
                location=r[rb]+r[ro];
                if(kind==0){width=4;load=0;} else if(kind==1){width=2;load=0;}
                else if(kind==2){width=1;load=0;} else if(kind==3){width=1;load=1;sign=1;}
                else if(kind==4){width=4;load=1;} else if(kind==5){width=2;load=1;}
                else if(kind==6){width=1;load=1;} else {width=2;load=1;sign=1;}
                if(sign&&width==2&&(location&1))width=1;
                int extra=0; Py_ssize_t size=0; uint32_t off=0; const unsigned char *mp=NULL;
                if(ram_region(location,width,&size,&off,&extra)) { mp=((location&0xff000000u)==0x02000000u)?(unsigned char*)ewv.buf:(unsigned char*)iwv.buf; if(width>1)off&=~(width-1); if(load){value=ram_read(mp,size,off,width);if(sign&&width==1&&value&0x80)value|=0xffffff00u;if(sign&&width==2&&value&0x8000)value|=0xffff0000u;r[rd]=value;base=3;}
                    else {ram_write((unsigned char*)mp,size,off,width,r[rd]);base=2;} total_cycles+=(uint32_t)extra;data_valid=1;data_addr=location&~(width-1);data_width=width; }
                else {uint32_t data_extra=0;if(bus_transfer(bus,ioreg,load,width,location,r[rd],&value,&data_extra)<0)goto fail;total_cycles+=data_extra;if(sign&&width==1&&value&0x80)value|=0xffffff00u;if(sign&&width==2&&value&0x8000)value|=0xffff0000u;if(load)r[rd]=value;base=load?3:2;}
            } else {
                unsigned rd=((op&0xf000)==0x9000)?((op>>8)&7):(op&7);
                if((op&0xe000)==0x6000){int byte=!!(op&0x1000);load=!!(op&0x0800);width=byte?1:4;location=r[(op>>3)&7]+(((op>>6)&31)<<(byte?0:2));}
                else if((op&0xf000)==0x8000){load=!!(op&0x0800);width=2;location=r[(op>>3)&7]+(((op>>6)&31)<<1);}
                else {load=!!(op&0x0800);width=4;location=r[13]+((op&255)<<2);}
                int extra=0; Py_ssize_t size=0; uint32_t off=0; const unsigned char *mp=NULL;
                if(ram_region(location,width,&size,&off,&extra)) { mp=((location&0xff000000u)==0x02000000u)?(unsigned char*)ewv.buf:(unsigned char*)iwv.buf; if(width>1)off&=~(width-1); if(load){value=ram_read(mp,size,off,width);if(width==4){unsigned rot=(location&3)*8;if(rot)value=(value>>rot)|(value<<(32-rot));}r[rd]=value;base=3;}
                    else {ram_write((unsigned char*)mp,size,off,width,r[rd]);base=2;} total_cycles+=(uint32_t)extra;data_valid=1;data_addr=location&~(width-1);data_width=width; }
                else {uint32_t data_extra=0;if(bus_transfer(bus,ioreg,load,width,location,r[rd],&value,&data_extra)<0)goto fail;total_cycles+=data_extra;if(load&&width==4){unsigned rot=(location&3)*8;if(rot)value=(value>>rot)|(value<<(32-rot));}if(load)r[rd]=value;base=load?3:2;}
            }
            if(!supported){stop=1;break;}
            }
            if(!supported){stop=1;break;}
        } else supported=0;
        if(!supported){stop=1;break;}
        int seq=prev_valid && prev+2==addr; total_cycles+=(uint32_t)(base+fetch_extra(addr,prev,seq,waitcnt));
        prev=addr;prev_valid=1;last_addr=addr;last_valid=1;r[15]=next;done++;
        pending_cycles+=total_cycles-cycles_before;
        if(pending_cycles>=(uint32_t)quantum){
            PyObject *updated=PyObject_CallMethod(bus,"advance_cycles","I",pending_cycles);
            if(!updated)goto fail;Py_DECREF(updated);pending_cycles=0;
            if(!(ps&0x80u) && (ioreg[0x208]&1) &&
               ((ioreg[0x200]|((uint16_t)ioreg[0x201]<<8)) & (ioreg[0x202]|((uint16_t)ioreg[0x203]<<8)))){stop=2;break;}
        }
        /* BX can leave Thumb state. Return control before interpreting the
           destination's ARM opcode as another 16-bit Thumb instruction. */
        if(!(ps&T)){stop=3;break;}
        /* Yield frequently so the GUI can observe IRQs and input state. */
    }
    for(int i=0;i<16;i++){PyObject *v=PyLong_FromUnsignedLong(r[i]);if(!v)goto fail;if(PyList_SetItem(regs_obj,i,v)<0)goto fail;}
    {PyObject *v=PyLong_FromUnsignedLong(ps);if(!v)goto fail;if(PyObject_SetAttrString(cpu,"cpsr",v)<0){Py_DECREF(v);goto fail;}Py_DECREF(v);}
    if(last_valid){PyObject *t=Py_BuildValue("(iiO)",(int)last_addr,2,Py_True);if(!t)goto fail;if(PyObject_SetAttrString(cpu,"_last_gamepak_fetch",t)<0){Py_DECREF(t);goto fail;}Py_DECREF(t);}
    Py_DECREF(regs_obj);PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);
    return Py_BuildValue("(iii)",done,(int)pending_cycles,stop);
fail:
    Py_DECREF(regs_obj);PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(ew);Py_DECREF(iw);Py_DECREF(bus);PyBuffer_Release(&rom);Py_DECREF(rom_obj);return NULL;
}
/* ARM compute/branch hot path. Memory transfers, exceptions, and privileged
   state operations deliberately return to the full Python semantic core. */
static PyObject *run_arm_batch(PyObject *self, PyObject *args) {
    PyObject *cpu=NULL,*regs=NULL,*bus=NULL,*cart=NULL,*rom_obj=NULL,*io=NULL,*ew=NULL,*iw=NULL;
    Py_buffer rom={0}, iov={0}, ewv={0}, iwv={0};
    int limit, quantum=256, initial_pending=0;
    if(!PyArg_ParseTuple(args,"Oi|ii",&cpu,&limit,&quantum,&initial_pending)) return NULL;
    if(limit<=0) return Py_BuildValue("(iii)",0,initial_pending,0);
    if(quantum<=0) quantum=256;
    PyObject *o=PyObject_GetAttrString(cpu,"cpsr"); if(!o)return NULL;
    uint32_t ps=(uint32_t)PyLong_AsUnsignedLongMask(o); Py_DECREF(o);
    if(ps&T || PyErr_Occurred()) {if(PyErr_Occurred())return NULL;return Py_BuildValue("(iii)",0,0,0);}
    bus=PyObject_GetAttrString(cpu,"bus"); if(!bus)return NULL;
    o=PyObject_GetAttrString(bus,"bios_data"); if(!o)goto fail;
    int hasbios=o!=Py_None; Py_DECREF(o); if(hasbios)goto done;
    cart=PyObject_GetAttrString(bus,"cartridge");if(!cart)goto fail;
    rom_obj=PyObject_GetAttrString(cart,"data");Py_DECREF(cart);cart=NULL;if(!rom_obj)goto fail;
    if(PyObject_GetBuffer(rom_obj,&rom,PyBUF_SIMPLE)<0)goto fail;
    io=PyObject_GetAttrString(bus,"io");if(!io)goto fail;
    if(PyObject_GetBuffer(io,&iov,PyBUF_SIMPLE)<0)goto fail;
    ew=PyObject_GetAttrString(bus,"ewram");if(!ew)goto fail;
    iw=PyObject_GetAttrString(bus,"iwram");if(!iw)goto fail;
    if(PyObject_GetBuffer(ew,&ewv,PyBUF_SIMPLE)<0)goto fail;
    if(PyObject_GetBuffer(iw,&iwv,PyBUF_SIMPLE)<0)goto fail;
    regs=PyObject_GetAttrString(cpu,"registers");if(!regs)goto fail;
    if(!PyList_Check(regs)||PyList_GET_SIZE(regs)<16||rom.len<4)goto done;
    uint32_t r[16];for(int i=0;i<16;i++){r[i]=(uint32_t)PyLong_AsUnsignedLongMask(PyList_GET_ITEM(regs,i));if(PyErr_Occurred())goto fail;}
    const unsigned char *code=(const unsigned char*)rom.buf;uint32_t cycles=(uint32_t)initial_pending;int done_count=0,stop=0;
    while(done_count<limit){
        uint32_t addr=r[15];if(addr<0x08000000u||addr>=0x0e000000u){stop=1;break;}
        uint32_t off=rom_offset(addr-0x08000000u,rom.len);if((Py_ssize_t)off+3>=rom.len){stop=1;break;}
        uint32_t ins=(uint32_t)code[off]|((uint32_t)code[off+1]<<8)|((uint32_t)code[off+2]<<16)|((uint32_t)code[off+3]<<24);
        if(!condition(ps,ins>>28)){r[15]=addr+4;cycles+=1;done_count++;}
        else if((ins&0x0ffffff0u)==0x012fff10u){uint32_t target=r[ins&15];r[14]=r[14];ps=(ps&~T)|(target&1?T:0);r[15]=target&((ps&T)?~1u:~3u);cycles+=3;done_count++;stop=2;}
        else if((ins&0x0e000000u)==0x0a000000u){int32_t delta=(int32_t)(ins<<8)>>6;if(ins&(1u<<24))r[14]=addr+4;r[15]=addr+8+(uint32_t)delta;cycles+=3;done_count++;}
        else if((ins&0x0f8000f0u)==0x00800090u){
            unsigned rdhi=(ins>>16)&15,rdlo=(ins>>12)&15,rs=(ins>>8)&15,rm=ins&15;
            int sign=!(ins&(1u<<22)),acc=!!(ins&(1u<<21)),set=!!(ins&(1u<<20));
            if(rdhi==15||rdlo==15||rs==15||rm==15||rdhi==rdlo){stop=1;break;}
            int64_t product=sign?(int64_t)(int32_t)r[rm]*(int64_t)(int32_t)r[rs]:(int64_t)((uint64_t)r[rm]*(uint64_t)r[rs]);
            uint64_t result=(uint64_t)product;if(acc)result+=((uint64_t)r[rdhi]<<32)|r[rdlo];
            r[rdlo]=(uint32_t)result;r[rdhi]=(uint32_t)(result>>32);
            if(set)ps=(ps&~(N|Z))|(r[rdhi]&N)|((result==0)?Z:0);
            cycles+=3;done_count++;r[15]=addr+4;
        }
        else if((ins&0x0fc000f0u)==0x00000090u){
            unsigned rd=(ins>>16)&15,rn=(ins>>12)&15,rs=(ins>>8)&15,rm=ins&15;
            int acc=!!(ins&(1u<<21)),set=!!(ins&(1u<<20));
            if(rd==15||rs==15||rm==15||(acc&&rn==15)){stop=1;break;}
            uint32_t result=(uint32_t)((uint64_t)r[rm]*(uint64_t)r[rs]);if(acc)result+=r[rn];
            r[rd]=result;if(set)nz(&ps,result);cycles+=2;done_count++;r[15]=addr+4;
        }
        else if((ins&0x0e000000u)==0x08000000u){
            int pre=!!(ins&(1u<<24)),up=!!(ins&(1u<<23)),user=!!(ins&(1u<<22)),wb=!!(ins&(1u<<21)),load=!!(ins&(1u<<20));
            unsigned rn=(ins>>16)&15,mask=ins&0xffffu,count=0;
            for(unsigned q=0;q<16;q++)if(mask&(1u<<q))count++;
            if(rn==15||!mask||user){stop=1;break;}
            uint32_t base=r[rn],span=count*4,start=up?base+(pre?4:0):base-span+(pre?0:4),final=up?base+span:base-span;
            unsigned char *ptrs[16];Py_ssize_t sizes[16];uint32_t offs[16];int waits[16];
            int valid=1;for(unsigned q=0,reg=0;q<16;q++)if(mask&(1u<<q)){
                uint32_t loc=start+reg*4;if(!ram_access(loc,4,(unsigned char*)ewv.buf,(unsigned char*)iwv.buf,&ptrs[reg],&sizes[reg],&offs[reg],&waits[reg])){valid=0;break;}offs[reg]&=~3u;reg++;}
            if(!valid){stop=1;break;}
            unsigned slot=0;for(unsigned reg=0;reg<16;reg++)if(mask&(1u<<reg)){
                cycles+=(uint32_t)waits[slot];
                if(load){uint32_t value=ram_read(ptrs[slot],sizes[slot],offs[slot],4);if(reg==15)r[15]=value&~3u;else r[reg]=value;}
                else {uint32_t value=reg==15?addr+12:r[reg];if(reg==rn&&reg!=(unsigned)__builtin_ctz(mask))value=final;ram_write(ptrs[slot],sizes[slot],offs[slot],4,value);}
                slot++;
            }
            if(wb&&!(load&&(mask&(1u<<rn))))r[rn]=final;
            if(!(load&&(mask&0x8000u)))r[15]=addr+4;
            cycles+=count+(load?2:1);done_count++;
        }
        else if((ins&0x0c000000u)==0x04000000u){
            int pre=!!(ins&(1u<<24)),up=!!(ins&(1u<<23)),byte=!!(ins&(1u<<22));
            int writeback=!!(ins&(1u<<21))||!pre,load=!!(ins&(1u<<20));
            unsigned rn=(ins>>16)&15,rd=(ins>>12)&15;uint32_t base=rn==15?addr+8:r[rn],offset;
            if(ins&(1u<<25)){
                if(ins&(1u<<4)){stop=1;break;}
                uint32_t value=(ins&15)==15?addr+8:r[ins&15];unsigned amount=(ins>>7)&31,type=(ins>>5)&3;
                if(type==0)offset=amount?value<<amount:value;
                else if(type==1)offset=amount?value>>amount:0;
                else if(type==2)offset=amount?(uint32_t)((int32_t)value>>amount):(uint32_t)((int32_t)value>>31);
                else offset=amount?((value>>amount)|(value<<(32-amount))):((value>>1)|((ps&C)?0x80000000u:0));
            } else offset=ins&0xfffu;
            uint32_t adjusted=up?base+offset:base-offset,location=pre?adjusted:base;
            if(rd==15 || (writeback&&(rn==15||(load&&rn==rd)))){stop=1;break;}
            Py_ssize_t size=0;uint32_t moff=0;int extra=0;unsigned width=byte?1:4;
            unsigned char *mem=NULL;
            if(!ram_access(location,width,(unsigned char*)ewv.buf,(unsigned char*)iwv.buf,&mem,&size,&moff,&extra)){stop=1;break;}
            uint32_t value;
            if(load){uint32_t aligned=moff&~3u;value=byte?ram_read(mem,size,moff,1):ram_read(mem,size,aligned,4);if(!byte&&(location&3)){unsigned rot=(location&3)*8;value=(value>>rot)|(value<<(32-rot));}r[rd]=value;}
            else {value=r[rd];if(rd==15)value=addr+12;ram_write(mem,size,byte?moff:(moff&~3u),width,value);}
            if(writeback)r[rn]=adjusted;
            cycles+=(uint32_t)(load?3:2)+(uint32_t)extra;done_count++;r[15]=addr+4;
        }
        else if((ins&0x0c000000u)==0){
            unsigned op=(ins>>21)&15, rn=(ins>>16)&15, rd=(ins>>12)&15;
            int setflags=!!(ins&(1u<<20)),writes=!(op>=8&&op<=11);
            if(rd==15 || (!setflags&&op>=8&&op<=11) || (ins&(1u<<25))==0&&(ins&(1u<<4))){stop=1;break;}
            uint32_t a=rn==15?addr+8:r[rn], b, shcarry=(ps&C)?C:0;
            if(ins&(1u<<25)){unsigned rot=((ins>>8)&15)*2;b=ins&255;b=rot?((b>>rot)|(b<<(32-rot))):b;if(rot)shcarry=(b&N)?C:0;}
            else {unsigned rm=ins&15,amount=(ins>>7)&31;b=rm==15?addr+8:r[rm];unsigned type=(ins>>5)&3;
                if(type==0){if(amount){shcarry=(b&(1u<<(32-amount)))?C:0;b<<=amount;}}
                else if(type==1){if(!amount){shcarry=(b&N)?C:0;b=0;}else{shcarry=(b&(1u<<(amount-1)))?C:0;b>>=amount;}}
                else if(type==2){if(!amount){shcarry=(b&N)?C:0;b=(b&N)?MASK32:0;}else{shcarry=(b&(1u<<(amount-1)))?C:0;b=(uint32_t)((int32_t)b>>amount);}}
                else if(!amount){shcarry=(b&1)?C:0;b=((b>>1)|((ps&C)?N:0));}
                else {shcarry=(b&(1u<<(amount-1)))?C:0;b=(b>>amount)|(b<<(32-amount));}}
            uint32_t res=0,lhs=a,rhs=b;int arithmetic=0,sub=0;uint64_t wide=0;
            unsigned carryin=!!(ps&C),borrow=1-carryin;
            switch(op){case 0:case 8:res=a&b;break;case 1:case 9:res=a^b;break;
                case 2:case 10:arithmetic=1;sub=1;wide=(uint64_t)a-(uint64_t)b;res=(uint32_t)wide;break;
                case 3:arithmetic=1;sub=1;lhs=b;rhs=a;wide=(uint64_t)b-(uint64_t)a;res=(uint32_t)wide;break;
                case 6:arithmetic=1;sub=1;wide=(uint64_t)a-(uint64_t)b-borrow;res=(uint32_t)wide;break;
                case 7:arithmetic=1;sub=1;lhs=b;rhs=a;wide=(uint64_t)b-(uint64_t)a-borrow;res=(uint32_t)wide;break;
                case 4:case 11:arithmetic=1;wide=(uint64_t)a+b;res=(uint32_t)wide;break;
                case 5:arithmetic=1;wide=(uint64_t)a+b+carryin;res=(uint32_t)wide;break;
                case 12:res=a|b;break;case 13:res=b;break;case 14:res=a&~b;break;default:res=~b;break;}
            if(writes)r[rd]=res;
            if(setflags){ps=(ps&~(N|Z|C|V))|(res&N)|(res==0?Z:0);
                if(arithmetic){int64_t sl=(int32_t)lhs,sr=(int32_t)rhs;
                    if(op==5)sr+=(int64_t)carryin;
                    int64_t exact=sub?sl-sr-((op==6||op==7)?(int64_t)borrow:0):sl+sr;
                    int overflow=exact>INT32_MAX||exact<INT32_MIN;
                    int carry=sub?(op==6?a>=(uint64_t)b+borrow:op==7?b>=(uint64_t)a+borrow:a>=b):(wide>MASK32);
                    if(carry)ps|=C;if(overflow)ps|=V;}
                else ps|=shcarry;}
            r[15]=addr+4;cycles+=1;done_count++;
        } else {stop=1;break;}
        if(cycles>=(uint32_t)quantum){o=PyObject_CallMethod(bus,"advance_cycles","I",cycles);if(!o)goto fail;Py_DECREF(o);cycles=0;}
        if(stop)break;
    }
    for(int i=0;i<16;i++){o=PyLong_FromUnsignedLong(r[i]);if(!o)goto fail;if(PyList_SetItem(regs,i,o)<0)goto fail;}
    o=PyLong_FromUnsignedLong(ps);if(!o)goto fail;if(PyObject_SetAttrString(cpu,"cpsr",o)<0){Py_DECREF(o);goto fail;}Py_DECREF(o);
    Py_DECREF(regs);PyBuffer_Release(&iwv);PyBuffer_Release(&ewv);Py_DECREF(iw);Py_DECREF(ew);PyBuffer_Release(&iov);Py_DECREF(io);PyBuffer_Release(&rom);Py_DECREF(rom_obj);Py_DECREF(bus);
    return Py_BuildValue("(iii)",done_count,(int)cycles,stop);
done:
    Py_XDECREF(regs);if(iwv.obj)PyBuffer_Release(&iwv);if(ewv.obj)PyBuffer_Release(&ewv);Py_XDECREF(iw);Py_XDECREF(ew);if(iov.obj)PyBuffer_Release(&iov);Py_XDECREF(io);if(rom.obj)PyBuffer_Release(&rom);Py_XDECREF(rom_obj);Py_XDECREF(cart);Py_XDECREF(bus);return Py_BuildValue("(iii)",0,0,0);
fail:
    Py_XDECREF(regs);if(iwv.obj)PyBuffer_Release(&iwv);if(ewv.obj)PyBuffer_Release(&ewv);Py_XDECREF(iw);Py_XDECREF(ew);if(iov.obj)PyBuffer_Release(&iov);Py_XDECREF(io);if(rom.obj)PyBuffer_Release(&rom);Py_XDECREF(rom_obj);Py_XDECREF(cart);Py_XDECREF(bus);return NULL;
}
/* Generate a run of PSG samples without calling four Python channel methods
   for every 32.768 kHz sample. Python still clocks the frame sequencer at its
   original 64-sample boundaries. */
static PyObject *run_psg_samples(PyObject *self, PyObject *args) {
    PyObject *psg,*io=NULL,*wave=NULL,*enabled=NULL,*volumes=NULL,*phases=NULL,*result=NULL,*o=NULL;
    Py_buffer iov={0},wv={0}; double phase[4],wave_pos; uint32_t lfsr; int frame_count;
    int count;
    if(!PyArg_ParseTuple(args,"Oi",&psg,&count))return NULL;
    if(count<=0)return PyList_New(0);
    io=PyObject_GetAttrString(psg,"io");wave=PyObject_GetAttrString(psg,"wave_ram");
    enabled=PyObject_GetAttrString(psg,"enabled");volumes=PyObject_GetAttrString(psg,"envelope_volume");phases=PyObject_GetAttrString(psg,"phase");
    if(!io||!wave||!enabled||!volumes||!phases)goto fail;
    if(PyObject_GetBuffer(io,&iov,PyBUF_SIMPLE)<0||PyObject_GetBuffer(wave,&wv,PyBUF_SIMPLE)<0)goto fail;
    if(!PyList_Check(enabled)||PyList_GET_SIZE(enabled)<4||!PyList_Check(volumes)||PyList_GET_SIZE(volumes)<4||!PyList_Check(phases)||PyList_GET_SIZE(phases)<4)goto fail;
    for(int i=0;i<4;i++){phase[i]=PyFloat_AsDouble(PyList_GET_ITEM(phases,i));if(PyErr_Occurred())goto fail;}
    o=PyObject_GetAttrString(psg,"wave_position");if(!o)goto fail;wave_pos=PyFloat_AsDouble(o);Py_DECREF(o);o=NULL;if(PyErr_Occurred())goto fail;
    o=PyObject_GetAttrString(psg,"noise_lfsr");if(!o)goto fail;lfsr=(uint32_t)PyLong_AsUnsignedLongMask(o);Py_DECREF(o);o=NULL;if(PyErr_Occurred())goto fail;
    o=PyObject_GetAttrString(psg,"frame_sample_count");if(!o)goto fail;frame_count=(int)PyLong_AsLong(o);Py_DECREF(o);o=NULL;if(PyErr_Occurred())goto fail;
    result=PyList_New(count);if(!result)goto fail;
    const unsigned char *regs=(const unsigned char*)iov.buf,*wave_bytes=(const unsigned char*)wv.buf;
    unsigned char active[4];int env[4];
    for(int c=0;c<4;c++){active[c]=(unsigned char)PyObject_IsTrue(PyList_GET_ITEM(enabled,c));env[c]=(int)PyLong_AsLong(PyList_GET_ITEM(volumes,c));if(PyErr_Occurred())goto fail;}
    static const double duty[4]={0.125,0.25,0.5,0.75};
    static const int low_reg[3]={0x64,0x6c,0x74},high_reg[3]={0x65,0x6d,0x75};
    for(int n=0;n<count;n++){
        if(++frame_count>=64){frame_count=0;o=PyObject_CallMethod(psg,"clock_frame_sequencer",NULL);if(!o)goto fail;Py_DECREF(o);o=NULL;
            for(int c=0;c<4;c++){active[c]=(unsigned char)PyObject_IsTrue(PyList_GET_ITEM(enabled,c));env[c]=(int)PyLong_AsLong(PyList_GET_ITEM(volumes,c));if(PyErr_Occurred())goto fail;}}
        int channel[4]={0,0,0,0};
        for(int c=0;c<2;c++)if(active[c]){unsigned reg=c?0x68:0x62;int freq=regs[low_reg[c]]|((regs[high_reg[c]]&7)<<8);
            phase[c]=fmod(phase[c]+131072.0/(2048-freq)/32768.0,1.0);channel[c]=(phase[c]<duty[regs[reg]>>6]?env[c]:-env[c])*128;}
        if(active[2]){int freq=regs[0x74]|((regs[0x75]&7)<<8);wave_pos=fmod(wave_pos+2097152.0/(2048-freq)/32768.0,64.0);
            unsigned dimension=!!(regs[0x70]&0x20),length=dimension?64:32,pos=(unsigned)wave_pos%length;
            unsigned bank=(!dimension&&((regs[0x70]>>6)&1))?16:0;unsigned packed=wave_bytes[bank+pos/2];int sample=(pos&1)?(packed&15):(packed>>4);
            unsigned volume_code=(regs[0x73]>>5)&3;double scale=(regs[0x73]&0x80)?0.75:(volume_code==0?0.0:volume_code==1?1.0:volume_code==2?0.5:0.25);
            channel[2]=(int)((sample-8)*scale*128.0);}
        if(active[3]){unsigned poly=regs[0x7c],ratio_code=poly&7,shift=(poly>>4)&15;double ratio=ratio_code?ratio_code:0.5;
            phase[3]+=524288.0/ratio/(double)(1u<<(shift+1))/32768.0;
            while(phase[3]>=1.0){phase[3]-=1.0;unsigned x=(lfsr^(lfsr>>1))&1;lfsr=(lfsr>>1)|(x<<14);if(poly&8)lfsr=(lfsr&~(1u<<6))|(x<<6);}
            channel[3]=(lfsr&1)?env[3]*128:-env[3]*128;}
        unsigned control=regs[0x80]|((unsigned)regs[0x81]<<8),ratio_code=regs[0x82]&3;double ratio=ratio_code==0?0.25:ratio_code==1?0.5:1.0;
        int left=0,right=0;for(int c=0;c<4;c++){if(control&(1u<<(12+c)))left+=channel[c];if(control&(1u<<(8+c)))right+=channel[c];}
        double lv=((control>>4)&7)/7.0,rv=(control&7)/7.0;
        PyObject *sample=Py_BuildValue("(ii)",(int)(left*lv*ratio),(int)(right*rv*ratio));if(!sample)goto fail;
        PyList_SET_ITEM(result,n,sample);
    }
    for(int i=0;i<4;i++){o=PyFloat_FromDouble(phase[i]);if(!o)goto fail;if(PyList_SetItem(phases,i,o)<0)goto fail;}
    o=PyFloat_FromDouble(wave_pos);if(!o)goto fail;if(PyObject_SetAttrString(psg,"wave_position",o)<0){Py_DECREF(o);goto fail;}Py_DECREF(o);o=NULL;
    o=PyLong_FromUnsignedLong(lfsr);if(!o)goto fail;if(PyObject_SetAttrString(psg,"noise_lfsr",o)<0){Py_DECREF(o);goto fail;}Py_DECREF(o);o=NULL;
    o=PyLong_FromLong(frame_count);if(!o)goto fail;if(PyObject_SetAttrString(psg,"frame_sample_count",o)<0){Py_DECREF(o);goto fail;}Py_DECREF(o);o=NULL;
    PyBuffer_Release(&wv);PyBuffer_Release(&iov);Py_DECREF(io);Py_DECREF(wave);Py_DECREF(enabled);Py_DECREF(volumes);Py_DECREF(phases);return result;
fail:
    Py_XDECREF(o);Py_XDECREF(result);if(wv.obj)PyBuffer_Release(&wv);if(iov.obj)PyBuffer_Release(&iov);Py_XDECREF(io);Py_XDECREF(wave);Py_XDECREF(enabled);Py_XDECREF(volumes);Py_XDECREF(phases);return NULL;
}
/* Fast renderer for ordinary mode-0 scenes. It handles tiled text BGs and
   regular (non-affine) OBJ sprites. Effects, windows, and mosaic stay on the
   reference Python path until their exact hardware rules are ported. */
static unsigned rgb555_component(unsigned value) {
    value &= 31u;
    return (value << 3) | (value >> 2);
}
static unsigned vram_index(unsigned offset) {
    offset &= 0x1ffffu;
    if (offset >= 0x18000u) offset -= 0x8000u;
    return offset;
}
static uint16_t video_u16(const unsigned char *bytes, unsigned offset) {
    return (uint16_t)(bytes[offset] | ((unsigned)bytes[offset + 1] << 8));
}
static void video_rgb(const unsigned char *palette, unsigned index, unsigned char *rgb) {
    unsigned color = video_u16(palette, (index & 255u) * 2u);
    rgb[0] = (unsigned char)rgb555_component(color);
    rgb[1] = (unsigned char)rgb555_component(color >> 5);
    rgb[2] = (unsigned char)rgb555_component(color >> 10);
}
static PyObject *render_mode0_fast(PyObject *self, PyObject *args) {
    PyObject *bus = NULL, *vram_obj = NULL, *palette_obj = NULL, *oam_obj = NULL, *io_obj = NULL;
    Py_buffer vram = {0}, palette = {0}, oam = {0}, io = {0};
    PyObject *result = NULL;
    unsigned char *pixels = NULL, *priorities = NULL;
    static const unsigned char shapes[3][4][2] = {
        {{8,8},{16,16},{32,32},{64,64}},
        {{16,8},{32,8},{32,16},{64,32}},
        {{8,16},{8,32},{16,32},{32,64}}
    };
    if (!PyArg_ParseTuple(args, "O", &bus)) return NULL;
    vram_obj = PyObject_GetAttrString(bus, "vram");
    palette_obj = PyObject_GetAttrString(bus, "palette_ram");
    oam_obj = PyObject_GetAttrString(bus, "oam");
    io_obj = PyObject_GetAttrString(bus, "io");
    if (!vram_obj || !palette_obj || !oam_obj || !io_obj) goto done;
    if (PyObject_GetBuffer(vram_obj, &vram, PyBUF_SIMPLE) < 0 ||
        PyObject_GetBuffer(palette_obj, &palette, PyBUF_SIMPLE) < 0 ||
        PyObject_GetBuffer(oam_obj, &oam, PyBUF_SIMPLE) < 0 ||
        PyObject_GetBuffer(io_obj, &io, PyBUF_SIMPLE) < 0) goto done;
    if (vram.len < 0x18000 || palette.len < 0x400 || oam.len < 0x400 || io.len < 0x56) goto done;

    const unsigned char *vr = (const unsigned char *)vram.buf;
    const unsigned char *pal = (const unsigned char *)palette.buf;
    const unsigned char *obj = (const unsigned char *)oam.buf;
    const unsigned char *regs = (const unsigned char *)io.buf;
    unsigned control = video_u16(regs, 0x000);
    if ((control & 7u) != 0 || (control & 0xe000u) ||
        (video_u16(regs, 0x050) & 0x00c0u)) goto done;

    /* This renderer deliberately declines unsupported per-pixel features. */
    for (unsigned bg = 0; bg < 4; ++bg) {
        if (!(control & (1u << (8u + bg)))) continue;
        unsigned bg_control = video_u16(regs, 0x008u + bg * 2u);
        if (bg_control & (1u << 6)) goto done;
    }
    if (control & (1u << 12)) {
        for (unsigned n = 0; n < 128; ++n) {
            unsigned attr0 = video_u16(obj, n * 8u);
            unsigned attr1 = video_u16(obj, n * 8u + 2u);
            unsigned attr2 = video_u16(obj, n * 8u + 4u);
            unsigned shape = attr0 >> 14, size = attr1 >> 14;
            unsigned mode = (attr0 >> 10) & 3u;
            if (mode >= 2 || shape == 3 || (!((attr0 >> 8) & 1u) && (attr0 & (1u << 9)))) continue;
            if ((attr0 & (1u << 8)) || (attr0 & (1u << 12))) goto done;
            (void)attr2;
        }
    }

    pixels = (unsigned char *)PyMem_Malloc(240u * 160u * 3u);
    priorities = (unsigned char *)PyMem_Malloc(240u * 160u);
    if (!pixels || !priorities) { PyErr_NoMemory(); goto done; }
    video_rgb(pal, 0, pixels);
    for (unsigned p = 0; p < 240u * 160u; ++p) {
        pixels[p * 3u] = pixels[0]; pixels[p * 3u + 1u] = pixels[1]; pixels[p * 3u + 2u] = pixels[2];
        priorities[p] = 4;
    }

    /* Paint BGs from farthest to nearest. This matches the Python compositor's
       priority ordering and keeps BG0 above BG1 above BG2 above BG3 on ties. */
    for (int priority = 3; priority >= 0; --priority) {
        for (int bg = 3; bg >= 0; --bg) {
            if (!(control & (1u << (8u + (unsigned)bg)))) continue;
            unsigned bgc = video_u16(regs, 0x008u + (unsigned)bg * 2u);
            if ((bgc & 3u) != (unsigned)priority) continue;
            unsigned char256 = (bgc >> 7) & 1u;
            unsigned char_base = ((bgc >> 2) & 3u) * 0x4000u;
            unsigned screen_base = ((bgc >> 8) & 31u) * 0x800u;
            unsigned size = bgc >> 14;
            unsigned width_tiles = (size == 1 || size == 3) ? 64u : 32u;
            unsigned height_tiles = (size == 2 || size == 3) ? 64u : 32u;
            unsigned width_pixels = width_tiles * 8u, height_pixels = height_tiles * 8u;
            unsigned hofs = video_u16(regs, 0x010u + (unsigned)bg * 4u) & 0x1ffu;
            unsigned vofs = video_u16(regs, 0x012u + (unsigned)bg * 4u) & 0x1ffu;
            unsigned tile_bytes = char256 ? 64u : 32u;
            for (unsigned y = 0; y < 160; ++y) {
                unsigned by = (y + vofs) % height_pixels;
                unsigned ty = by >> 3, py = by & 7u;
                for (unsigned x = 0; x < 240; ++x) {
                    unsigned bx = (x + hofs) % width_pixels;
                    unsigned tx = bx >> 3, px = bx & 7u;
                    unsigned source_y = py;
                    unsigned block = (ty >> 5) * (width_tiles >> 5) + (tx >> 5);
                    unsigned map = screen_base + block * 0x800u + (((ty & 31u) * 32u + (tx & 31u)) * 2u);
                    unsigned entry = vr[vram_index(map)] | ((unsigned)vr[vram_index(map + 1u)] << 8);
                    if (entry & (1u << 10)) px = 7u - px;
                    if (entry & (1u << 11)) source_y = 7u - source_y;
                    unsigned tile = char_base + (entry & 0x3ffu) * tile_bytes;
                    unsigned palette_index;
                    if (char256) {
                        palette_index = vr[vram_index(tile + source_y * 8u + px)];
                    } else {
                        unsigned packed = vr[vram_index(tile + source_y * 4u + (px >> 1))];
                        unsigned color = (packed >> ((px & 1u) * 4u)) & 15u;
                        if (!color) continue;
                        palette_index = (((entry >> 12) & 15u) * 16u) + color;
                    }
                    if (!palette_index) continue;
                    unsigned pixel = y * 240u + x;
                    video_rgb(pal, palette_index, pixels + pixel * 3u);
                    priorities[pixel] = (unsigned char)priority;
                }
            }
        }
    }

    /* OBJ priority ties sit over BGs; among equal-priority OBJ pixels, lower
       OAM indices win. Clip rectangles before entering the pixel loops. */
    if (control & (1u << 12)) {
        unsigned mapping_1d = (control >> 6) & 1u;
        for (int priority = 3; priority >= 0; --priority) {
            for (int n = 127; n >= 0; --n) {
                unsigned off = (unsigned)n * 8u;
                unsigned attr0 = video_u16(obj, off), attr1 = video_u16(obj, off + 2u), attr2 = video_u16(obj, off + 4u);
                unsigned mode = (attr0 >> 10) & 3u, shape = attr0 >> 14, size = attr1 >> 14;
                if (mode >= 2 || shape == 3 || (!((attr0 >> 8) & 1u) && (attr0 & (1u << 9)))) continue;
                if (((attr2 >> 10) & 3u) != (unsigned)priority) continue;
                unsigned width = shapes[shape][size][0], height = shapes[shape][size][1];
                int x = (int)(attr1 & 0x1ffu), y = (int)(attr0 & 0xffu);
                if (x >= 256) x -= 512;
                if (y >= 160) y -= 256;
                int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
                int x1 = x + (int)width, y1 = y + (int)height;
                if (x1 > 240) x1 = 240;
                if (y1 > 160) y1 = 160;
                if (x0 >= x1 || y0 >= y1) continue;
                unsigned color256 = (attr0 >> 13) & 1u;
                unsigned hflip = (attr1 >> 12) & 1u, vflip = (attr1 >> 13) & 1u;
                unsigned tile = attr2 & 0x3ffu, tile_unit = color256 ? 2u : 1u;
                if (color256) tile &= ~1u;
                unsigned width_tiles = width >> 3, bank = (attr2 >> 12) & 15u;
                for (int sy = y0; sy < y1; ++sy) {
                    unsigned local_y = (unsigned)(sy - y), source_y = vflip ? height - 1u - local_y : local_y;
                    unsigned tile_y = source_y >> 3, py = source_y & 7u;
                    for (int sx = x0; sx < x1; ++sx) {
                        unsigned pixel = (unsigned)sy * 240u + (unsigned)sx;
                        if ((unsigned)priority > priorities[pixel]) continue;
                        unsigned local_x = (unsigned)(sx - x), source_x = hflip ? width - 1u - local_x : local_x;
                        unsigned tile_x = source_x >> 3, px = source_x & 7u;
                        unsigned tile_offset = mapping_1d ? tile_y * width_tiles * tile_unit + tile_x * tile_unit : tile_y * 32u + tile_x * tile_unit;
                        unsigned address = 0x10000u + (tile + tile_offset) * 32u;
                        unsigned palette_index;
                        if (color256) palette_index = vr[vram_index(address + py * 8u + px)];
                        else {
                            unsigned packed = vr[vram_index(address + py * 4u + (px >> 1))];
                            unsigned color = (packed >> ((px & 1u) * 4u)) & 15u;
                            if (!color) continue;
                            palette_index = bank * 16u + color;
                        }
                        if (!palette_index) continue;
                        video_rgb(pal + 0x200u, palette_index, pixels + pixel * 3u);
                        priorities[pixel] = (unsigned char)priority;
                    }
                }
            }
        }
    }
    result = PyBytes_FromStringAndSize((const char *)pixels, 240 * 160 * 3);
done:
    if (vram.obj) PyBuffer_Release(&vram);
    if (palette.obj) PyBuffer_Release(&palette);
    if (oam.obj) PyBuffer_Release(&oam);
    if (io.obj) PyBuffer_Release(&io);
    Py_XDECREF(vram_obj); Py_XDECREF(palette_obj); Py_XDECREF(oam_obj); Py_XDECREF(io_obj);
    PyMem_Free(pixels); PyMem_Free(priorities);
    if (!result && !PyErr_Occurred()) Py_RETURN_NONE;
    return result;
}
static PyMethodDef methods[]={{"run_thumb_batch",run_thumb_batch,METH_VARARGS,"Execute a batch of safe Thumb instructions in native C."},{"run_arm_batch",run_arm_batch,METH_VARARGS,"Execute safe ARM compute and branch instructions in native C."},{"run_psg_samples",run_psg_samples,METH_VARARGS,"Generate PSG audio samples in native C."},{"render_mode0_fast",render_mode0_fast,METH_VARARGS,"Render ordinary mode-0 tiled backgrounds and sprites in native C."},{NULL,NULL,0,NULL}};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_native_cpu",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native_cpu(void){return PyModule_Create(&module);}
