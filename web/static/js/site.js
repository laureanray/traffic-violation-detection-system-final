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
        console.log('get latest data');
        $("#car_count").html(data.cars_detected);
        $("#violation_count").html(data.violations_detected)
        console.log(data);
    });

    getLatestViolations(function(data){
      console.log('Get Latest');
      initDataTable(data);
    });

    // getViolations(function(data){
    //   data = JSON.parse(data);
    // })
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

function getLatestViolations(callback){
  $.ajax({
    type: 'GET',
    dataType: 'json',
    url: '/get_violations',
    success: function(res){
      callback(res);
    },
    error: function(err){
      console.log(err);
    }
  })
}

$(document).ready(function(){
  console.log('Doc is ready');
 
  getLatestViolations(function(data){
    initDataTable(data);
  });
});

function initDataTable(data){

for(var i = 0; i < data.length; i++){
  data[i].plate_number_img = "<a> <img class='img-fluid' style='height: 30px!important' src='http://localhost:8080/" + data[i].plate_number_img_url +"'></a>";
  data[i].car_img = "<a> <img class='img-fluid' style='height: 60px!important'  src='http://localhost:8080/" + data[i].vehicle_img_url +"'></a>";
  data[i].date = moment(data[i].date).format('LLL');
}

console.log(data);

$("#dataTable").DataTable({
      "destroy": true,
      "data": data,
      "columns": [
        { "data": "violation_type" },
        { "data": "plate_number_img" },
        { "data": "car_img" },
        { "data": "date" }
      ] 
    });

}



function getLatestData(callback){
  $.ajax({
    url: "./get_global",
    type: "GET",
    success: function(data){
      callback(data)
    },
    error: function(err){
      console.log(err);
      // alert(err);
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

