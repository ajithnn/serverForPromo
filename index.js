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
        fs.readFile('/app/assets/out.txt', 'utf8', function(err, data) {
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

rest.get('/runpy', function(err, query, ctype) {
    var statFlag = true;
    if (!err) {
        var pathforPython = 'python ';
        var pathForFile = __dirname + '/python/grayFaceGreenEye.py '; // Change here the python file name.
        console.log(pathforPython + pathForFile + " /app/assets/outImage_" + query.id + ".txt"); // change the params below also.
        exec(pathforPython + pathForFile + " /app/assets/outImage_" + query.id + ".txt", function(error, stdout, stderr) {
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
var time = new Date("January 1 1970 00:00:00");
var OldTime = null;
io.on('connection', function(socket) {
    console.log("Connected: " + socket.id);
    fs.watch('./assets/', function(event, filename) {
        if (filename == "out.txt") {
            OldTime = time;
            time = new Date().getTime();
            if(time - OldTime < 150)
            {
            console.log("Sent");
            fs.readFile('/app/assets/out.txt', 'utf8', function(err, data) {
                if (err) {
                    return console.log(err);
                }
                io.to(socket.id).emit("OpenUrl", data);
            });
            }
        } else if (filename == "error_" + socket.id) {
            fs.readFile(filename, 'utf8', function(err, data) {
                if (err) {
                    return console.log(err);
                }
                io.to(socket.id).emit("Error", data);
            });
        }
    });
});
rest.port = process.env.PORT || 3000;
rest.listen(process.env.PORT || 3000, function() {
    console.log("Listening on port 0.0.0.0:%s", process.env.PORT || rest.port)
})