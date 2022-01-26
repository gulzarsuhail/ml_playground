const PADDING_X = 30;
const PADDING_Y = 30;
// all tensor values are scaled as per max grid size
const PRESUME_MAX_GRID_SIZE = 1500

let enableMouseTicks = true;
let soundEnable = true;

const Graph = {

    /*
        Draws the x and y axis along the left and bottom of the screen
        @args
        sx: padding from the left of the screen
        sy: padding ffrom the bottom of the screen
    */
    drawAxis: function (sx, sy) {
        stroke(210);
        strokeWeight(1.5);
        // for y-axis
        line(sx, 0, sx, windowHeight);
        // for x-axis
        line(0, windowHeight - sy, windowWidth, windowHeight - sy);
    },

    /*
    Draws the grid on the screen along with the tick labels.
    @config
    gridSize variable controls the size of a box of the grid.
    @args
    sx: origin x axis
    sy: origin y axis
    */
    drawGrid: function (sx, sy) {
        const gridSize = 50;
        const numOfXticks = floor((windowHeight - sy) / gridSize);
        const numOfYticks = floor((windowWidth - sy) / gridSize);
        stroke(50);
        strokeWeight(1);
        fill(200, 200, 200);
        textAlign(CENTER);
        // draw x ticks
        for (let i = numOfXticks + 1; i > -1; i--) {
            line(0, windowHeight - sy - (i * gridSize), windowWidth, windowHeight - sy - (i * gridSize));
            if (i > 0 && i % 2 == 0) {
                strokeWeight(0);
                text(50 * i, sx / 2, windowHeight - sy - (i * gridSize) + 4);
                strokeWeight(1);
            }
        }
        // draw y ticks
        for (let i = -1; i < numOfYticks + 1; i++) {
            line(sx + (i * gridSize), 0, sx + (i * gridSize), windowHeight);
            if (i > 0 && i % 2 == 0) {
                strokeWeight(0);
                text(50 * i, (i * gridSize) + sy, windowHeight - PADDING_Y + 20);
                strokeWeight(1);
            }
        }
    },

    /*
        Draws the grid and axes on the screen
        @args
        sx: padding of axes from left
        sy: padding of axes from bottom
    */
    draw: function (sx, sy) {
        this.drawGrid(sx, sy);
        this.drawAxis(sx, sy);
    }
}

/*
    Used to create a new instance of a point on screen
    @args:
    x: points x axis
    y: points y axis
*/
function DataPoint(x, y) {
    this.x = x;
    this.y = y;

    // nimationStatus 1 means animation start and 0 means the end
    this.animationStats = 1;

    // Shows the dot on the screen
    this.showOnScreen = function () {
        circle(this.x, this.y, 10);
        // if animation of the dot is not completed, animate the dot
        if (this.animationStats > 0) {
            this.animationStats -= 0.025;
            fill('rgba(250, 255, 145, ' + this.animationStats + ')');
            circle(this.x, this.y, this.animationStats * 50);
        }
    }
}

// all the dots are stored here
const DataSet = {
    x_vals: [],
    y_vals: [],
    dataPoints: [],

    // preload the sound if sound is enabled
    loadSound: function (soundEnabled) {
        this.soundEnabled = soundEnabled;
        if (soundEnabled) {
            soundFormats('mp3');
            this.newDotSound = loadSound('sound_assets/new_dot');
        }
    },

    playSound: function () {
        if (this.soundEnabled) {
            this.newDotSound.play();
        }
    },

    // adds new datapoint and crets a new instance of DataPoint for the same
    addNewData: function (x, y, maxGridSize) {
        this.x_vals.push(map(x, 0, maxGridSize, 0, 1));
        this.y_vals.push(map(y, 0, maxGridSize, 1, 0));
        this.dataPoints.push(new DataPoint(x, y));
        this.playSound(this.newDotSound);
    },

    // removes off screen dots when the screen is resized
    removeOffScreenDots: function () {
        for (let i = this.dataPoints.length - 1; i >= 0; i--) {
            if (this.dataPoints[i].x > windowWidth || this.dataPoints[i].y > windowHeight) {
                this.x_vals.splice(i, 1);
                this.y_vals.splice(i, 1);
                this.dataPoints.splice(i, 1);
            }
        }
    },

    // to show dots on screen, runs showOnScreen for all dots
    drawDataPoints: function () {
        fill(245, 242, 66);
        for (let i = 0; i < this.dataPoints.length; i++) {
            this.dataPoints[i].showOnScreen();
        }
    },

    // removes all data
    clearAllData: function () {
        this.x_vals = [];
        this.y_vals = [];
        this.dataPoints = [];
    }
}

const LinearRegression = {

    optimizer: tf.train.sgd(0.5),
    lastLoss: 0,
    training: true,
    animationStateLineColor: 0,

    init: function (m_random, c_random) {
        this.m = tf.variable(tf.scalar(m_random)),
            this.c = tf.variable(tf.scalar(c_random))
    },
    predict: function (X) {
        const x_ten = tf.tensor1d(X);
        return x_ten.mul(this.m).add(this.c);

    },
    loss: function (pred, lablels) {
        const ssr = pred.sub(lablels).square().mean();
        const newLoss = ssr.dataSync()[0].toFixed(9);
        if (newLoss == this.lastLoss){
            this.training = false;
        }
        this.lastLoss = newLoss;
        return ssr;
    },


    /*
        Trains the model and shows best fit line on screen
        @args
        X: the predictors
        y: the target
        maxGridSize: the best fit line will be scaled accordingly
        @returns
        if still training
    */ 
    train: function (X, y, maxGridSize) {
        if (this.training){
            this.optimizer.minimize(() =>
                this.loss(this.predict(X), tf.tensor1d(y))
            );
        }
        this.drawBestFitLine(maxGridSize);
        return this.training;
    },

    /*
        Draws the best fit line on screen
        @args
        maxGridSize: The max size of the presumed grid, used to scale all coordiates
    */
    drawBestFitLine: function (maxGridSize) {
        strokeWeight(2);
        if (this.training){
            stroke(255);
            const X = [0, windowWidth / maxGridSize];
            const y = this.predict(X).dataSync();
            this.lineCoordinates  = [
                map(X[0], 0, 1, 0, maxGridSize),
                map(y[0], 0, 1, maxGridSize, 0),
                map(X[1], 0, 1, 0, maxGridSize),
                map(y[1], 0, 1, maxGridSize, 0)
            ]
        } else {
            stroke('rgb(171,245,174)');
        }
        line( ...this.lineCoordinates );
    },

    enableTraining: function(){
        this.training = true;
    }
}

// Draws the ticks from the axes to the mouse location
function drawMouseTicks() {
    if (enableMouseTicks) {
        stroke(255);
        strokeWeight(0.5);
        line(PADDING_X, mouseY, mouseX, mouseY);
        line(mouseX, mouseY, mouseX, windowHeight - PADDING_Y);
    }
}

// Is executed automatically if window is resized
function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
    // remove dots which are now off screen after resize
    DataSet.removeOffScreenDots();
}

// Is executed automatically only once on page load
function setup() {
    DataSet.loadSound(soundEnable);
    LinearRegression.init(random(1), random(1));
    createCanvas(windowWidth, windowHeight);
}

/*
    Is executed automatically when mouse button is pressed.
    Only exists because if mousePressed is not present, touchStated
    is executed by p5 automatically.
*/
function mousePressed() {
    return false;
}

/*
    Is executed automatically when mouse button is released.
    Only exists because if mouseReleased is not present, touchEnded
    is executed by p5 automatically.
*/
function mouseReleased() {
    enableMouseTicks = true;
    return false;
}

// On mouse click add a new data point
function mouseClicked() {
    LinearRegression.enableTraining();
    DataSet.addNewData(mouseX, mouseY, PRESUME_MAX_GRID_SIZE);
    return false;
}

/*
    If on a touch device, only show mouceTicks when the finger touches the screen
    Thus only enables mouse ticks when touch starts
*/
function touchStarted() {
    enableMouseTicks = true;
    return false;
}

/*
    ~ Complimentary to above function.
    If on a touch device, only show mouceTicks when the finger touches the screen
    Thus diables mouse ticks when touch ends
*/
function touchEnded() {
    LinearRegression.enableTraining();
    DataSet.addNewData(mouseX, mouseY, PRESUME_MAX_GRID_SIZE);
    enableMouseTicks = false;
    return false;
}


/*
    This function is executed 60 times a second (60FPS) automatically by p5.
    Used to redraw the whole screen screen on each execution.
*/
function draw() {
    background('#222');
    Graph.draw(PADDING_X, PADDING_Y);
    drawMouseTicks();
    if (DataSet.x_vals.length > 0) {
        DataSet.drawDataPoints();
        // to prevent memory leak tidy up the ununsed tensors
        tf.tidy(() => {
            LinearRegression.train(DataSet.x_vals, DataSet.y_vals, PRESUME_MAX_GRID_SIZE);
        });
    }
}

/*
    Set up on reset button press event listner
*/

window.onload = function () {
    const reset_button = document.getElementById('reset');
    reset_button.addEventListener('click', (event) => {
        // setTimeout(()=> DataSet.clearAllData(), 20);
        event.stopPropagation();
        DataSet.clearAllData();
    });
}