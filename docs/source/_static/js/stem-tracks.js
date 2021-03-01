// Available arguments can be found in the following page.
// https://github.com/naomiaro/waveform-playlist/#playlist-options
// https://github.com/naomiaro/waveform-playlist/#track-options

var playlist = WaveformPlaylist.init({
    samplesPerPixel: 5000,
    waveHeight: 100,
    container: document.getElementById("collage-results"),
    timescale: true,
    state: 'cursor',
    colors: {
        waveOutlineColor: '#E0EFF1'
    },
    controls: {
        show: true, //whether or not to include the track controls
        width: 180 //width of controls in pixels
    },
    zoomLevels: [1000, 3000, 5000],
    exclSolo: true,
    isAutomaticScroll: true
});

playlist.load([{
        "src": "../_audio/collage.mp3",
        "name": "Original Song",
        "gain": 0.5
    },
    {
        "src": "../_audio/collage_vocal_contour.mp3",
        "name": "Vocal Contour",
        "gain": 1
    },
    {
        "src": "../_audio/collage_vocal.mp3",
        "name": "Vocal",
        "gain": 1
    }
])