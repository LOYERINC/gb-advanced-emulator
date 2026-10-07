# GBA Emulator Implementation Plan

The goal is to build a small emulator from scratch, learning the hardware as we go. We will make one change at a time and keep the project runnable at each milestone. A milestone marked **Done** means the code exists; it does not imply the whole GBA is emulated accurately yet.

## Current project

- **Done:** Load a `.gba` file and read its title, game code, maker code, version, and header checksum.
- **Done:** First-pass memory arrays and byte/halfword/word bus access.
- **Done:** A Tkinter desktop window can open a ROM and display its metadata.
- **In progress:** CPU registers, condition flags, ARM control flow, math, multiply, swap, halfword/byte/word and block transfers, plus an initial Thumb instruction subset. ARM models banked registers/SPSRs for User/System, FIQ, IRQ, Supervisor, Abort, and Undefined modes; LDM/STM user-bank transfers, `LDM^` exception return, ARMv4T empty-list PC transfers with ±`0x40` writeback, and ARMv4 base-in-list writeback behavior are present. Thumb LDM/STM now handles ARMv4T empty register lists and the special writeback rules when the base appears in the list. ARM and Thumb loads model ARM7TDMI's unaligned word/halfword results; ARM register-shift operands that read PC use the core's PC+12 pipeline value. Immediate ARM data-processing opcodes no longer get misclassified as multiply/halfword instructions just because their literal has certain low bits; this fixed the observed `BIC r3, r3, #0xDF` stop at PC `0x03003688`. Direct boot initializes the System stack to `0x03007F00`, sets `POSTFLG` to its post-BIOS value, and starts the LCD at VCOUNT `0x7E`, matching the cartridge handoff point used by mGBA's BIOS-skip path; IRQ and Supervisor modes get stacks at `0x03007FA0` and `0x03007FE0`. Common BIOS calls (math including ArcTan/ArcTan2, BIOS checksum, MidiKey2Freq, BIOS sound-driver call stubs, BG/OBJ affine matrix setup, memory copy, LZ77/RL/Huffman decompress, BitUnPack, Diff8/Diff16 unfilter, SoftReset, approximate HardReset, RegisterRamReset, Halt/CustomHalt, keypad-woken Stop, IntrWait, and VBlankIntrWait) have high-level replacements. Writes to HALTCNT also request Halt or Stop at the next CPU instruction boundary. MultiBoot returns failure when link transfer is unavailable; SoundGetJumpList and MusicPlayerOpen/Start/Stop/Continue/FadeOut SWIs are recognized silent stubs. The UI can optionally load a user-supplied 16 KiB BIOS and start from its reset vector; SWI exceptions then enter its handler. BIOS reads are gated by the CPU execution address and expose an approximate last-opcode open-bus value outside BIOS. Immediate DMA copy/fill and a timed-DMA trigger interface are present.
- **In progress:** Video renderer draws Mode 0 tiled backgrounds, affine tiled backgrounds in Modes 1 and 2, regular and affine sprites, and bitmap modes 3, 4, and 5. It tracks the top two visible layers and applies WIN0/WIN1/OBJ-window clipping, alpha blending, brightness increase/decrease, semi-transparent OBJ blending, BG/OBJ mosaic, and DISPCNT forced blank. Bitmap fast paths now fall back to the full compositor whenever BLDCNT enables color effects; direct Mode 3/4/5 scenes confirm brightening matches the full compositor and changes the image. Ordinary Mode 0 text backgrounds cache decoded tile rows and composite enabled backgrounds in priority order without per-pixel layer bookkeeping. The full compositor also uses segmented, cached tile rows when BG mosaic is off, preserving windows, sprites, and color effects; direct runtime comparisons match the original per-pixel renderer for patterned 4bpp and 8bpp scenes with one, two, and four layers, scroll, tile flips, windows, and effects. A one-layer scene with OBJ enabled took 108 ms through the cached path versus 285 ms through the per-pixel path. Affine tile rendering caches map entries, decoded rows, and the palette for each frame; eight direct Mode 1/2 scene comparisons matched the prior per-pixel renderer across scaling, rotation, wrapping, mosaic, and priority. Sprite rendering now reuses decoded tile rows and object-palette colors across visible OBJ pixels; direct 4bpp/8bpp checks confirmed normal and horizontally flipped pixels. OBJ-window rendering now also reuses decoded 4bpp/8bpp tile rows; direct mask checks confirmed transparent texels and horizontal flipping. Mode 1/2 text and affine backgrounds are sorted together by priority, with lower-numbered BGs winning ties; direct pixel checks cover higher/lower priorities and ties. A previous single-layer measurement was 62–70 ms through the simple fast path versus 290–315 ms through the original compositor. Comparisons also exposed and fixed vertical-flip and transparent-pixel bugs.
- **In progress:** Run/pause controls execute supported instruction batches, and keyboard input is captured across the application so widget focus does not block game controls; key events update the active-low GBA keypad register. Timer registers support separate reload/current values, prescalers, cascade mode, and overflow IRQ flags. A rough-cycle LCD scheduler updates HBlank/VBlank/VCOUNT, sets enabled display IRQ flags, and triggers HBlank/VBlank DMA. The common no-LCD-edge case now skips the display boundary loop; 20,000 varied cycle advances matched the original event path, and a focused 360,000-call timing comparison was about 23% faster. Thumb conditional branches now distinguish taken from not-taken timing, and the first half of a long branch-link gets its own estimate. Cycle estimates now count only Thumb's actual register-list bits for `PUSH`, `POP`, `LDMIA`, and `STMIA`, rather than counting opcode bits as extra registers. ARM and Thumb multiply operations now report their operand-dependent ARM7TDMI cycle counts to timers and the LCD scheduler. Ordinary CPU operations now account for both instruction fetch and execution, with separate estimates for PSR transfers, swaps, loads/stores, and ARM/Thumb block transfers. ARM/Thumb instruction fetches from the three Game Pak ROM windows use WAITCNT nonsequential/sequential settings, including the two 16-bit halves of an ARM fetch and forced nonsequential timing at each 128 KiB boundary. CPU data accesses also add WAITCNT-based Game Pak/SRAM and first-pass EWRAM/16-bit-memory waits, and one visible-draw wait for palette/VRAM accesses. Exact video contention and OAM locking remain approximate. The Game Pak prefetch buffer, DMA bus timing, and some region-specific access details remain unmodeled. Built-in CPU copy, fill, bit-unpack, and decompression BIOS replacements now advance timers and display according to output work instead of charging every whole routine as a three-cycle call. Halt and interrupt-wait idle steps advance hardware in 32-cycle quanta rather than polling one emulated cycle per Python step; this keeps waits responsive while reducing empty CPU work, with up to 31 cycles of interrupt wake-up delay. Audio advances its sample remainder directly and calls the mixer only at a sample boundary; enabled sample output matched the always-mix path for varied cycle advances. The CPU can enter IRQ mode, use banked IRQ SP/LR and SPSR, and return with an exception-return instruction; a small locally implemented BIOS-style dispatcher calls the IWRAM callback at `0x03007FFC` without bundling BIOS firmware.
- **Known limits:** Most hardware registers are still plain storage, several memory regions are simplified, many CPU instructions and BIOS calls are unsupported, and hardware-accurate mixing/timing remain incomplete. WAITCNT and CPU data bus timing are approximated; video-memory contention is only modeled as one visible-draw wait, while OAM access restrictions, Game Pak prefetch, and DMA bus timing remain unmodeled. IRQ entry is implemented; generic undefined opcodes enter the user BIOS undefined vector when a BIOS is loaded, while FIQ, prefetch-abort, and data-abort entry are not implemented. Optional BIOS read-protection open-bus timing and post-BIOS startup state are approximate. STOP currently wakes on keypad IRQ only; cartridge and serial wake sources are absent. KEYCNT can request a keypad IRQ through the shared interrupt path, though timing and interrupt-handler compatibility remain approximate. Direct Sound FIFO A/B and first-pass PSG pulse/wave/noise channels can produce approximate Windows audio; SRAM, Flash, EEPROM, and RTC-marked cartridges are recognized. The RTC uses the host computer's local clock, and non-RTC cartridge GPIO devices are unsupported. If a ROM lacks a save marker, the desktop UI allows a user who knows its hardware to choose SRAM, Flash, Flash 1M, or EEPROM manually. LCD and audio events use estimated CPU cycles.

## Milestones

### 1. Make cartridge loading dependable — first pass complete

- Reject empty or too-small files with a useful message.
- Parse the rest of the fixed cartridge header and report malformed fields clearly.
- Keep the ROM as read-only data and expose its declared size and header checksum.
- **Done when:** valid ROMs show sensible metadata; bad files fail without a traceback.
- **Current status:** Empty and undersized files are rejected. Header fixed-byte/checksum problems are reported as warnings so homebrew and diagnostic files can still be inspected.

### 2. Refine the memory bus

- Keep all memory-region sizes and address ranges in one documented place.
- Check alignment and little-endian behavior for byte, halfword, and word accesses. **First pass:** byte writes repeat across BG VRAM/palette halfwords, ignore OBJ VRAM/OAM, and SRAM/Flash halfword/word accesses follow the cartridge's 8-bit bus behavior.
- Handle ROM windows, RAM mirrors, video-memory mirroring, and unmapped reads deliberately.
- Replace plain IO bytes with register reads/writes as each device is added.
- Implement DMA channel registers and transfer timing; **first pass:** immediate DMA works. The LCD scheduler triggers VBlank/HBlank DMA, and timer overflows can trigger Direct Sound FIFO special DMA. DMA bus-cycle costs and more specialized channels remain approximate.
- **Done when:** small bus checks cover region boundaries, mirrors, and read-only ROM behavior.

### 3. Build the ARM7TDMI CPU core

- Add CPU registers, status flags, processor modes, and banked registers. **First pass:** base registers and condition flags are present; banked modes remain.
- Add instruction fetch and condition checks. **First pass:** ARM-state fetch, conditions, B/BL, BX, common data-processing operations including register-controlled shifts, and single byte/word transfers are present. Several special cases remain.
- Implement ARM instructions in groups: branches, data processing, loads/stores, then multiply and status-register instructions. **First pass:** branches, common data-processing operations, multiply, swap, LDR/STR byte/halfword/word, block transfers including user-bank forms and exception return via `LDM^ ... PC`, and common CPSR/SPSR MRS/MSR forms are present. IRQ entry works with the local dispatcher, and undefined-instruction entry works when a user BIOS is loaded; FIQ/abort entry, timing, and many instruction edge cases remain.
- Implement Thumb instructions after the ARM path is stable. **First pass:** common Thumb arithmetic, logic, branches, stack operations, and byte/halfword/word memory transfers are present; exception handling and edge cases remain.
- Model exceptions and the instruction pipeline where games depend on them.
- **Done when:** public CPU test ROMs pass for the implemented ARM and Thumb groups.

### 4. Start a game without Nintendo firmware

- Do not include copyrighted BIOS data in the project.
- Allow users to load an optional 16 KiB BIOS image they are entitled to use, or implement the small set of high-level BIOS calls needed by games. **First pass:** the UI accepts a user-selected image and SWI exceptions enter Supervisor mode at the BIOS vector; SoftReset, approximate HardReset, RegisterRamReset, Halt/CustomHalt, Div, DivArm, Sqrt, ArcTan, ArcTan2, GetBiosChecksum, MidiKey2Freq, SoundBias, SoundDriverInit, SoundChannelClear, SoundDriverMode/Main/VSync, SoundGetJumpList, MultiBoot failure, MusicPlayerOpen/Start/Stop/Continue/FadeOut stubs, BgAffineSet, ObjAffineSet, CpuSet, CpuFastSet, BitUnPack, Diff8/Diff16 unfilter, HuffUnComp, LZ77UnCompWram/Vram, RLUnCompWram/Vram, IntrWait, and VBlankIntrWait are emulated when no BIOS image is selected. BIOS read protection, complete startup setup, and the MPlay mixer remain incomplete.
- Define a documented direct-boot path for development and initialize CPU and memory consistently.
- **Done when:** a legally usable test program reaches its entry point and can execute simple code.
- **Verified example:** `examples/gradient_demo.gba` has a valid cartridge header, branches to its ARM entry code, and fills all 240x160 Mode 3 pixels without a BIOS image.

### 5. Add controller input and a basic display

- Map buttons to the GBA keypad input register. **First pass:** keyboard mappings update active-low KEYINPUT, while KEYCNT button selection and AND/OR conditions can request keypad IRQ 12.
- Create a window showing the native 240 by 160 screen.
- Start with bitmap display mode 3, then add tiled backgrounds and sprites. **First pass:** bitmap modes 3, 4, and 5, text/affine tiled backgrounds, and regular/affine sprites render from VRAM, OAM, and palette RAM. WIN0/WIN1 clip layers and special effects by their programmed rectangles; non-transparent OBJ-window pixels use the high WINOUT settings. Alpha blending, brightness effects, and semi-transparent sprites use BLDCNT/BLDALPHA/BLDY. BG/OBJ mosaic reads its block sizes from MOSAIC and honors each layer/sprite's enable bit.
- Add display timing and VBlank status so games can synchronize to frames.
- **Done when:** a small homebrew demo can draw an image and react to buttons.
- **Current status:** Keyboard-to-keypad mapping is present; rendering supports Mode 0 and 1/2 affine backgrounds, bitmap modes 3-5, normal/affine sprites, WIN0/WIN1/OBJ-window, first-pass color effects, and BG/OBJ mosaic. The included Mode 3 homebrew displays a full color pattern. Frame synchronization remains.

### 6. Add timing and hardware services

- Implement interrupts, timers, and DMA, connected to CPU cycle counts. **First pass:** timer reload/current/control registers, 1/64/256/1024 prescalers, cascade counting, timer overflow IF flags, IF write-one-to-clear behavior, rough instruction-cycle estimates, LCD scanline/VCOUNT/VBlank/HBlank flags, enabled display IRQ requests, VBlank/HBlank DMA triggers, IRQ mode/register banking, SPSR-based exception return, and a local BIOS-style IRQ dispatcher. Other processor modes and accurate timing remain.
- Add scanline/frame scheduling and memory-access timing as needed for compatibility.
- **Done when:** test programs using these features behave predictably over repeated frames.

### 7. Add sound and save data

- Implement the GBA sound channels and sample output. **First pass:** 32-byte Direct Sound A/B FIFOs, SOUNDCNT routing/volume/timer selection, FIFO reset, channel 1/2 sound-DMA refills, four PSG channels (pulse, sweep, wave, noise), envelope/length/frame-sequencer behavior, the documented noise-counter frequency formula, stereo mixing, and an optional Windows PCM playback thread. Sound-driver init is only a compatibility stub; its full MPlay software mixer and DMA work area are not implemented. More accurate mixing/timing and audio verification with test ROMs remain.
- Detect common save types and persist save data in a project save folder. **First pass:** ROM marker detection, persistent 32 KiB SRAM, command-driven 64/128 KiB Flash, and 512 B/8 KiB EEPROM serial reads/writes through DMA3; saves flush on eject or application close.
- **Done when:** homebrew audio and save tests work across emulator restarts.

### 8. Improve compatibility and usability

- Run open homebrew and emulator test ROMs; fix regressions as features are added.
- Add pause, reset, frame pacing, key configuration, and useful error messages.
- Profile slow parts and optimize only after correctness is established.
- **First pass:** The desktop UI has Open ROM, Step, Run, Pause, Reset game, and Eject controls. Reset restarts the loaded cartridge and flushes its save data.
- **Done when:** a documented set of test programs works reliably and the emulator is comfortable to use.

## How we will work

1. Pick the next small milestone or sub-step.
2. Implement it in the existing Python project.
3. Explain what changed and what limitation comes next.
4. Run focused checks when requested, using homebrew or public test programs where possible.

Commercial games may need many rounds of hardware-accuracy work after the first homebrew program runs. The project will not ship game ROMs or Nintendo BIOS files.
