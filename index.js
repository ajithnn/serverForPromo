var rest = require('exprestify')
var fs = require('fs')
var exec = require('child_process').exec;

var header = {
    "Access-Control-Allow-Origin": "null",
    "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Credentials": "true"
};

var multiopt = {
    FilePath: "./assets/",
    PostType: "file",
    Rename: function(fld, file) {
        return "CurrentImg";
    }
};

rest.setHeaders(header);

rest.getfile('/image', function(err, query) {
    if (!err) {
        fs.readFile("./assets/" + query.id + ".txt", 'utf8', function (err,data) {
                if (err) {
                        return console.log(err);
                }
                       return data;
                });
    } else {
        console.log(err);
        return err;
    }
})

rest.get('/runpy', function(err, query,ctype) {
    var statFlag = true;
    if (!err) {
        var pathforPython = 'python ';
        var pathForFile = __dirname + '/python/grayFaceGreenEye.py ';// Change here the python file name.
        console.log(pathforPython + pathForFile + __dirname + " outImage_"+ query.id + ".jpg");// change the params below also.
        exec(pathforPython + pathForFile + __dirname + " outImage_"+ query.id + ".txt" + " error_" + query.id, function(error, stdout, stderr) {
             console.log(stdout);  
             console.log(error); 
        });
        return "done"
    } else {
        console.log(err);
        return err;
    }
})

rest.multipost('/PostPhoto', function(err, data) {
    if (!err) {
        console.log(data);
        return "done";
    } else {
        console.log(err);
    }
}, multiopt);

var io = rest.getSocketServer()

io.on('connection', function(socket) {
	console.log("Connected: " + socket.id);
    fs.watch('./assets/', function(event, filename) {
        if(filename == "outImage_"+socket.id+".jpg")
        {
        var time = new Date().getTime();
        console.log("Sent");
        io.to(socket.id).emit("OpenUrl", "/image?id=outImage_" + socket.id + "&time=" + time);
        }
	else if(filename == "error_" + socket.id){
		fs.readFile(filename, 'utf8', function (err,data) {
 	 	if (err) {
    			return console.log(err);
  		}
  			io.to(socket.id).emit("Error",data);
		});
	}
    });
});
rest.port = process.env.PORT || 3000 ;
rest.listen(process.env.PORT || 3000, function() {
    console.log("Listening on port 0.0.0.0:%s", process.env.PORT || rest.port)
})
