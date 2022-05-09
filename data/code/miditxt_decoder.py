from mido import MidiFile, MidiTrack, Message, MetaMessage
from miditxt_encoder import *

def channel_assemble(data, ch):
    trackdata = data[ch]

    typ = trackdata.typ.split()
    meta = trackdata.meta.split('\n')
    program = meta[1].split() 
    cc_c = trackdata.cc_control.split()
    cc_t = trackdata.cc_time.split()
    cc_v = trackdata.cc_value.split()
    no_n = trackdata.no_note.split()
    no_t = trackdata.no_time.split()
    no_v = trackdata.no_velocity.split()
    pw_p = trackdata.pw_pitch.split()
    pw_t = trackdata.pw_time.split()

    noindx, ccindx, pwindx = 0, 0, 0
    track = MidiTrack()
    if program: 
        m = Message("program_change", channel = ch, program=int(program[0]), time=int(program[1]))
        track.append(m)
    for node in typ:
        if node == "no":
            m = Message("note_on", channel = ch, note=int(no_n[noindx]), time=int(no_t[noindx]), velocity=int(no_v[noindx]))
            noindx += 1
        elif node == "cc":
            m = Message("control_change", channel = ch, control=int(cc_c[ccindx]), time=int(cc_t[ccindx]), value=int(cc_v[ccindx]))
            ccindx += 1
        elif node == "pw":
            m=Message("pitchwheel", channel = ch, pitch=int(pw_p[pwindx]), time=int(pw_t[pwindx]))
            pwindx += 1
        else:
            print("Unknown node", node)
        track.append(m)
        
    return track

def postprocessing(data, lentrack, speed=120, tempo = 810810, nom = 4, denom = 4, k = 'C'):

    newmid = MidiFile()
    newmid.ticks_per_beat=speed
    metatrack = MidiTrack()
    metatrack.append(MetaMessage("set_tempo", tempo=int(tempo)))
    metatrack.append(MetaMessage("time_signature", numerator = int(nom), denominator = int(denom)))
    metatrack.append(MetaMessage("key_signature", key=k))
    newmid.tracks.append(metatrack)

    for i in range(lentrack):
        newmid.tracks.append(channel_assemble(data, i))

    newmid.save("output.mid")



if __name__ == "__main__":

    postprocessing(*preprocess())


