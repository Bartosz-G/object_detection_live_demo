let init = async () => {

    localStream = await navigator.mediaDevices.getUserMedia({video: true, audio: false})
    document.getElementById('viewer').srcObject = localStream

}

