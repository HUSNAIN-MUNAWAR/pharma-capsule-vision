const video = document.getElementById("inspectionVideo");
const playButton = document.getElementById("playButton");
const restartButton = document.getElementById("restartButton");
const videoFrame = document.querySelector(".video-frame");
const videoState = document.getElementById("videoState");
const progressFill = document.getElementById("progressFill");
const timecode = document.getElementById("timecode");
const clock = document.getElementById("clock");

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00";
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const remainder = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remainder}`;
}

function updateVideoUi() {
  const duration = video.duration || 0;
  const progress = duration ? (video.currentTime / duration) * 100 : 0;
  progressFill.style.width = `${progress}%`;
  timecode.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
  if (video.ended) videoState.textContent = "RUN COMPLETE";
  else if (video.paused) videoState.textContent = video.currentTime ? "PAUSED" : "READY TO PLAY";
  else videoState.textContent = "PLAYING FULL RUN";
  videoFrame.classList.toggle("is-playing", !video.paused && !video.ended);
}

function toggleVideo() {
  if (video.paused || video.ended) video.play();
  else video.pause();
}

playButton.addEventListener("click", toggleVideo);
video.addEventListener("click", toggleVideo);
restartButton.addEventListener("click", () => { video.currentTime = 0; video.play(); });
["loadedmetadata", "timeupdate", "play", "pause", "ended"].forEach((eventName) => video.addEventListener(eventName, updateVideoUi));

function updateClock() {
  const now = new Date();
  clock.dateTime = now.toISOString();
  clock.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

updateClock();
setInterval(updateClock, 1000);
