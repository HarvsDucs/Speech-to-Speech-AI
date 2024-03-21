import replicate
from groq import Groq
import os
from dotenv import load_dotenv
import json
import requests
from pydub import AudioSegment
from playsound import playsound
import pyaudio
import numpy as np
import wave
import time
import elevenlabs

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
groq_api_key = os.getenv("groq_api_key")



audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
frames = []

try:
    while True:
        data = stream.read(1024)
        frames.append(data)

except KeyboardInterrupt:
    pass

stream.stop_stream()
stream.close()
audio.terminate()

sound_file = wave.open("recording.wav", "wb")
sound_file.setnchannels(1)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(44100)
sound_file.writeframes(b''.join(frames))
sound_file.close()

print("Proceeding")
start_time = time.time()
input = {
    "audio": open("recording.wav", "rb"),
    "batch_size": 64
}

output = replicate.run(
    "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
    input=input
)

#=> {"text":" the little tales they tell are false the door w...

# Convert the text output to a JSON string
text_output_json = json.dumps(output)

# Parse the JSON string back to a Python dictionary
parsed_output = json.loads(text_output_json)

# Access specific elements from the parsed output
text = parsed_output['text']

#print(text)

end_time = time.time()
execution_time = end_time - start_time
print("Faster Whisper Execution time in seconds:", execution_time)

start_time = time.time()

client = Groq(
    api_key=groq_api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": text + "Limit your reply to 30 words max",
        }
    ],
    model="mixtral-8x7b-32768",
)


#print(chat_completion.choices[0].message.content)
AI_reply = chat_completion.choices[0].message.content
print(AI_reply)
print("AI Reply done")

end_time = time.time()
execution_time = end_time - start_time

print("Groq Execution time in seconds:", execution_time)
#input = {
#    "text": "hello, how are you?",
#    "embedding_scale": 1.5
#}
#
#output = replicate.run(
#    "adirik/styletts2:989cb5ea6d2401314eb30685740cb9f6fd1c9001b8940659b406f952837ab5ac",
#    input=input
#)
#print(output)

print(AI_reply)


start_time = time.time()
url = "https://api.elevenlabs.io/v1/text-to-speech/P7x743VjyZEOihNNygQ9/stream"

payload = {"text": AI_reply}
headers = {
    "xi-api-key": "7142734f8f58d82a23aa0518a74d4281",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

with open("response_audio.mp3", "wb") as file:
    file.write(response.content)

# Play the audio file

end_time = time.time()
execution_time = end_time - start_time
print("Elevenlabs time in seconds:", execution_time)

playsound("response_audio.mp3")
