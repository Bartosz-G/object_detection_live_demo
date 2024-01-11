
let init = async () => {
    try {
        let localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        document.getElementById('viewer').srcObject = localStream;
    } catch (error) {
        console.error('Error accessing media devices.', error);
    }
};

init();
