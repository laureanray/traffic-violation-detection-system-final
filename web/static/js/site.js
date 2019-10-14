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
        $("#violation_count").html(data.violations_detected)
    });

    getViolations(function(data){
      data = JSON.parse(data);
    })
    // getViolations(function(data){
    //   data = JSON.parse(data);
    //   $("#dataTable").DataTable({
    //     data: data,
    //     columns: [
    //       { data: 'violation_type' },
    //       { data: 'vehicle_type' },
    //       { data: 'plate_number' },
    //       { data: 'plate_number' }
    //     ],
    //   })
    // })
});

// $("#dataTable").DataTable();



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

function getViolations(callback){
  $.ajax({
    url: "./get_violations",
    type: "GET",
    contentType: "application/json",
    success: function(data){
      callback(data);
    },
    error: function(err){
      alert(err);
    }
  })
}

