const stunServers = [
    {
        urls: ['stun:stun.kinesisvideo.eu-west-2.amazonaws.com:443', 'stun.l.google.com']
    }
];

const ip = '13.42.36.213';
const offerEndpoint = '/offer';

let localStream;
let remoteStream;
let peerConnection;


let websocket;
var client_id = Date.now();
let websocket_url = 'ws://' + window.location.hostname + `:80/ws/${client_id}`;




let sendMessage = async (message) => {
    if (websocket.readyState === WebSocket.OPEN) {
        websocket.send(message);
    } else {
        console.error('WebSocket is not open. ReadyState: ', websocket.readyState);
    }
}


let createOffer = async () => {
    let offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer);


    let message = JSON.stringify({'type': 'offer', 'offer': offer})

    console.log("Local Offer set:", message);

    return message
}


let createAnswer = async () => {
    let answer = await peerConnection.createAnswer()
    await peerConnection.setLocalDescription(answer);

    sdp = {'sdp': peer_connection.localDescription};
    sendMessage(JSON.stringify(sdp));
}


let handleIncomingSDP = async ( sdp ) => {
    peerConnection.setRemoteDescription(sdp).then(() => {
        console.log('Successfully set Remote description!');
        if (sdp.type != 'offer') {
            return;
        } else {
            peerConnection.setLocalDescription(sdp)
                .then(createAnswer)
                .catch((e) => {console.error('Error while creating answer: ' + e)});
        }
    }).catch((e) => {console.error('Error while setting Remote Description: ' + e)});
}

let handleIncomingICE = async ( ice ) => {
    var candidate = new RTCIceCandidate(ice);
    peerConnection.addIceCandidate(candidate).catch((e) => {console.error('Error while handling incoming ICE: ' + e)});
}

let onIceCandidate = ( event ) => {
    if (event.candidate == null) {
        console.log("ICE Candidate was null, done with gethering ICE candidates");
        return;
    };

    sendMessage(JSON.stringify({'ice': event.candidate}));
}




let onWebsocketMessage = async (event) => {
    console.log('Received a message from websocket of type: ' + event.data)

    if (event.data === 'PONG') {
        console.log(event.data)
        return;
    }

    if (event.data === 'waitingForOffer') {
        let Offer = await createOffer();
        sendMessage(Offer);
        return;
    }

    try {
        msg = JSON.parse(event.data);
    } catch (e) {
        console.error('Failed to parse a message from the websocket' + event.data);
    }

    if (msg.sdp != null) {
        handleIncomingSDP(msg.sdp);
    } else if (msg.ice != null) {
        console.log('Received new ICE candidate: ', msg.ice)
        handleIncomingICE(msg.ice);
    } else {
        console.error('Unknown message type')
    }


}

let onWebsocketError = () => {} // Impl Websocket error handling

let onWebsocketClose = () => {} // Impl Socket Close handling

let websocketConnect = () => {} // Refactor later



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

    peerConnection = new RTCPeerConnection( stunServers );



    localStream.getTracks().forEach((track) => {
            peerConnection.addTrack(track, localStream)
        });

    peerConnection.onicecandidate = onIceCandidate;

    // ------- Monitoring ICE negotiations ---------
    peerConnection.onicecandidateerror = e => {
        console.log("ERROR: ICE candidate Error: ", e)
    }

    peerConnection.oniceconnectionstatechange = () => {
            console.log("ICE connection state change:", peerConnection.iceConnectionState);
    }

    peerConnection.onicegatheringstatechange = () => {
        console.log("ICE gathering state change:", peerConnection.iceGatheringState);
    };
    // -----------


    try {
        websocket = new WebSocket(websocket_url)
        websocket.onmessage = onWebsocketMessage;
        websocket.onopen = (e) => {
            sendMessage("PING");
        };
    } catch (error) {
        console.error('Error accessing the websocket', error)
    }

};










init();
