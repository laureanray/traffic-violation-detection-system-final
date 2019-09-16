var socket = io.connect('http://' + document.domain + ':' + location.port);
socket.on('connect', function() {
    socket.emit('connected');
    getLatestData(function(data){
      $("#car_count").html(data.cars_detected);
  });
});
socket.on('update', function(data) {
    // $('h1').text(data)
    console.log("May update");
    getLatestData(function(data){
        $("#car_count").html(data.cars_detected);
    });
});


function getLatestData(callback){
  $.ajax({
    url: "./get_global",
    type: "GET",
    success: function(data){
      callback(data)
    },
    error: function(err){
      // console.log(err);
      alert(err);
    }
  })
}

