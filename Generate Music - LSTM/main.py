# 1. prepareing the data
import glob
from music21 import converter, instrument, note, chord, stream
import numpy as np
from keras.utils import to_categorical

notes = []

for file in glob.glob("Generate Music - LSTM/midi_files/pop songs/*.mid"):
    # load each file into a Music21 stream object
    # midi = a list of all the notes and chords in the file
    midi = converter.parse(file)
    notes_to_parse = None

    # parts = instrument.partitionByInstrument(midi)

    # if parts: # file has instrument parts
    #     notes_to_parse = parts.parts[0].recurse()
    # else:
    #     notes_to_parse = midi.flat.notes
    notes_to_parse = midi.flat.notes
    # we want notes and chords as the input and output
    for element in notes_to_parse:
        # print(element)
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))


# 1.2 string -> integer
# NN perform better w/ integer-based data than string-based data
# put the length of each seq to be 100 notes/chords
# to predict the next note, it has the previous 100 notes to help make the prediction
#### HIGHLY RECOMMEND use different lengths for prediction
sequence_length = 100

# get all pitch names
pitchnames = sorted(set(item for item in notes))
n_vocab = len(set(notes))

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))


# 1.3 create input & output sequences for NN
# output for each input seq will be the first note/chord that comes after the sequence of notes in the input sequence in our list of notes
network_input = []
network_output = []

# create input seq and the corr outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i : i+sequence_length]
    sequence_out = notes[i+sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
n_patterns = len(network_input)

# reshape the input into a format compatible w/ LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
# normalize input
network_input = network_input / float(n_vocab)

network_output = to_categorical(network_output)



# 2. Training Model
# LSTM layers
# Dropout layers
# Dense layers / fully connected layers
# Activation layer
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

#### HIGHLY RECOMMEND play w/ the structure
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
# optimizer: RMSprop - good choice for recurrent NN
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
filepath = "weights_pop.hdf5"
# stop running the NN once we are satistied w/ the loss value
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=0, save_best_only=True, mode='min'
)
callbacks_list = [checkpoint]
model.fit(network_input, network_output, epochs=5, batch_size=64, callbacks=callbacks_list)



# 3. Generating Model
model = Sequential()
model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# load the weights to each node
model.load_weights('weights_pop.hdf5')



# 4. Generate music
start = np.random.randint(0, len(network_input)-1)

int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

pattern = network_input[start]
prediction_output = []

# generate 500 notes
for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)
    prediction = model.predict(prediction_input, verbose=0)

    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]



# 5. Decode the output
# Chord - split the string up into an array of notes
#         loop through the string representation of each note and create a Note obj for each
# Node - create a Note obj useing the string representation of the pitch
# increase the offset by 0.5 at the end of each iteration
offset = 0
output_notes = []

for pattern in prediction_output:
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern is a note
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
    # increase offset each iter so that notes do not stack    
    offset += 0.5



# 6. create music!
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='pop_test_output.mid')