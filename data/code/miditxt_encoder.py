from mido import MidiFile, MidiTrack, Message
from data import *

def preprocess(filename =  'HotelCalifornia.mid'):
    data = []
    mid = MidiFile(filename)
    speed = mid.ticks_per_beat  


    for n, track in enumerate(mid.tracks[1:]):
        typ = ''

        no_note = ''
        no_velocity = ''
        no_time = ''
        lno = 0

        cc_control = ''
        cc_value = ''
        cc_time = ''
        lcc = 0

        pw_pitch = ''
        pw_time = ''
        lpw = 0

        program_change = ''

        for i in track:
            if i.type == "note_on":
                lno += 1
                typ += 'no'+' '
                no_note += str(i.note)+' '
                no_velocity += str(i.velocity)+' '
                no_time += str(i.time)+ ' '
            elif i.type == "control_change":
                lcc += 1
                typ += 'cc'+' '
                cc_control += str(i.control)+' '
                cc_value += str(i.value)+' '
                cc_time += str(i.time)+' '
                #print(i)
            elif i.type == "program_change":
                program_change += str(i.program)+' '+str(i.time)+"\n"
                #print('i')
            elif i.type == "pitchwheel":
                lpw += 1
                typ += 'pw'+' '
                pw_pitch += str(i.pitch)+' '
                pw_time += str(i.time)+' '
            elif i.is_meta:
                pass
                #print(i)
            else:
                print(i, "at track", n)

        for i in mid.tracks[0]:
            if i.type == "set_tempo":
                tempo = i.tempo
            elif i.type == "time_signature":
                nom = i.numerator
                denom = i.denominator
            elif i.type == "key_signature":
                key = i.key

        msg = MessageData()
        msg.ch = n
        msg.typ = typ
        msg.no_note = no_note
        msg.no_velocity = no_velocity
        msg.no_time = no_time
        msg.cc_control = cc_control
        msg.cc_value = cc_value
        msg.cc_time = cc_time
        msg.pw_pitch = pw_pitch
        msg.pw_time = pw_time
        msg.meta = f"{lno+lcc} {lno} {lcc}\n" + program_change

        data.append(msg)

        
    return data, len(mid.tracks)-1, speed, tempo, nom, denom, key


if __name__ == "__main__":
    preprocess()

