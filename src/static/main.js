const iceServers = [
    {
        urls: ['stun:stun.kinesisvideo.eu-west-2.amazonaws.com:443', 'stun.l.google.com']
    }
];

const ip = '13.42.36.213';
const offerEndpoint = '/offer';

let localStream;
let remoteStream;
let peerConnection;

let init = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('MediaDevices API not supported or blocked in this browser.');
        return;
    }
    try {
        localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
        document.getElementById('viewer').srcObject = localStream;
    } catch (error) {
        console.error('Error accessing media devices.', error);
    }

    createOffer()
};

let fetchOffer = async (serverEndpoint, message) => {
    fetch(serverEndpoint, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: message})
        .then(response => response.json())
        .then(data => {console.log('Success:', data);})
        .catch((error) => {console.error('Error:', error);});

}

let createOffer = async () => {
    peerConnection = new RTCPeerConnection( iceServers );

    localStream.getTracks().forEach((track) => {
            peerConnection.addTrack(track, localStream)
        });

    let offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer);


    let message = JSON.stringify({'type': 'offer', 'offer': offer})

    console.log("Local Offer:", message);

    fetchOffer(offerEndpoint, message);


}


init();
