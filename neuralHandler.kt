import java.io.File
import java.io.BufferedReader
import kotlin.math.E
import kotlin.math.pow

fun main() {
        // Load ANN weights and input ranges
        var bufferedReader: BufferedReader 


        bufferedReader = File("raw/input_to_hidden_weights_and_biases").bufferedReader()
        val inputToHiddenWeightsAndBiasesString = bufferedReader.use {it.readText()}

        bufferedReader = File("raw/hidden_to_output_weights_and_biases").bufferedReader()
        val hiddenToOutputWeightsAndBiasesString = bufferedReader.use {it.readText()}

        bufferedReader = File("raw/input_ranges").bufferedReader()
        val inputRangesString = bufferedReader.use {it.readText()}


        val numWindowsBatched = 1

        var nh = NeuralHandler(
            inputToHiddenWeightsAndBiasesString,
            hiddenToOutputWeightsAndBiasesString,
            inputRangesString,
            numWindowsBatched
        )

        nh.processBatch()
}

class NeuralHandler (inputToHiddenWeightsAndBiasesString: String,
                     hiddenToOutputWeightsAndBiasesString: String,
                     inputRangesString:String,
                     private var numWindows: Int){

    private var inputToHiddenWeightsAndBiases: Matrix
    private var hiddenToOutputWeightsAndBiases: Matrix
    private var inputRanges: Matrix
    private val windowSize = 100
    private var state = 0
    private var sessionState = 0
    private var currentPuffLength = 0
    private var currentInterPuffIntervalLength = 0

    init{
        inputToHiddenWeightsAndBiases = Matrix(inputToHiddenWeightsAndBiasesString)
        hiddenToOutputWeightsAndBiases = Matrix(hiddenToOutputWeightsAndBiasesString)
        inputRanges = Matrix(inputRangesString)
        if (numWindows < 1) {
            throw IllegalArgumentException("Number of Windows Batched must be 1 or greater")
        }
    }

    fun processBatch(extrasBuffer: MutableList<MutableList<String>>,
                     xBuffer: MutableList<MutableList<Double>>,
                     yBuffer: MutableList<MutableList<Double>>,
                     zBuffer: MutableList<MutableList<Double>>,
                     fRaw: FileOutputStream) {
        /*
            extrasBuffer: 3x200 timestamps and activity with each row:
                            [SensorEvent timestamp (ns), Calendar timestamp (ms), current activity]
            xBuffer:    1x200 x-axis accelerometer data values
            yBuffer:    1x200 y-axis accelerometer data values
            zBuffer:    1x200 z-axis accelerometer data values
            fRaw:       File output stream to write accelerometer raw data

            This function calls forwardPropagate for each of the windows
            (for 100 windows: [0-100...99-199] in the input matrix and then writes the data points
            and ANN outputs to a file
        */

        var smokingOutput: Double

        // Run ANN on windows
        var i = 0
        while(i < numWindows){
            smokingOutput = forwardPropagate(
                Matrix((xBuffer.slice(i until i+windowSize)+
                        yBuffer.slice(i until i+windowSize)+
                        zBuffer.slice(i until i+windowSize)).toMutableList()))
            if (smokingOutput >= 0.85){
                smokingOutput = 1.0
            }
            else{
                smokingOutput = 0.0
            }
            // puff counter
            if (state == 0 && smokingOutput == 0.0){
                // no action
            } else if (state == 0 && smokingOutput == 1.0){
                // starting validating puff length
                state = 1
                currentPuffLength ++
            } else if (state == 1 && smokingOutput == 1.0){
                // continuing not yet valid length puff
                currentPuffLength ++
                if (currentPuffLength > 14) {
                    // valid puff length!
                    state = 2
                }
            } else if (state == 1 && smokingOutput == 0.0){
                // never was a puff, begin validating end
                state = 3
                currentInterPuffIntervalLength ++
            } else if (state == 2 && smokingOutput == 1.0){
                // continuing already valid puff
                currentPuffLength ++
            } else if (state == 2 && smokingOutput == 0.0){
                // ending already valid puff length
                state = 4 // begin validating inter puff interval
                currentInterPuffIntervalLength ++
            } else if (state == 3 && smokingOutput == 0.0) {
                currentInterPuffIntervalLength ++
                if (currentInterPuffIntervalLength > 49){
                    // valid interpuff
                    state = 0
                    currentPuffLength = 0
                    currentInterPuffIntervalLength = 0
                }
            } else if (state == 3 && smokingOutput == 1.0){
                // was validating interpuff for puff that wasn't valid
                currentPuffLength ++
                currentInterPuffIntervalLength = 0
                if (currentPuffLength > 14) {
                    // valid puff length!
                    state = 2
                }
                state = 1
            } else if (state == 4 && smokingOutput == 0.0) {
                currentInterPuffIntervalLength ++
                if (currentInterPuffIntervalLength > 49){
                    // valid interpuff for valid puff
                    state = 0
                    currentPuffLength = 0
                    currentInterPuffIntervalLength = 0
                    onPuffDetected()
                }
            } else if (state == 4 && smokingOutput == 1.0){
                // back into puff for already valid puff
                currentInterPuffIntervalLength = 0
                currentPuffLength ++
                state = 2
            }
//            Log.d("0000","state $state")
            // fRaw.write((extrasBuffer[i][0]+","+
            //         xBuffer[i][0]+","+
            //         yBuffer[i][0]+","+
            //         zBuffer[i][0]+","+
            //         extrasBuffer[i][1]+","+
            //         extrasBuffer[i][2]+","+
            //         smokingOutput.toString()+","+
            //         state.toString()+"\n").toByteArray())
            i++
        }
    }
    private fun onPuffDetected(){
        // write to file
    }
    private fun forwardPropagate(input: Matrix): Double {
        /*
            input : three-axis accelerometer values from smartwatch sampled at 20 Hz for 5 seconds.
                    i.e. the input is (1x300) where the first 100 values are x accelerometer
                    readings, the second 100 values are y accelerometer readings, and the last
                    are z accelerometer readings. The vector sort of looks like this:
                            [x,x,x,x,...,x,y,y,y,...,y,z,z,z,...,z].
                    If anyone asks where this info is found, see (Cole et al. 2017) at
                    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5745355/.
         */
        var normedInput = minMaxNorm(input)
        normedInput.addOneToFront()
        var hiddenOutput = tanSigmoid(inputToHiddenWeightsAndBiases * normedInput)
        hiddenOutput.addOneToFront()
        return logSigmoid(hiddenToOutputWeightsAndBiases * hiddenOutput)[0][0]
    }
}

class Matrix {
    private var mRowSize: Int = 0
    private var mColSize: Int = 0
    private var mData: MutableList<MutableList<Double>> = mutableListOf()
    var shape: Pair<Int,Int> = Pair(0,0)

    constructor(rowSize: Int, colSize: Int){
        mRowSize = rowSize
        mColSize = colSize
        shape = Pair(rowSize,colSize)
        mData = MutableList(mRowSize) { MutableList(mColSize) { 0.0 } }
    }
    constructor(data: MutableList<MutableList<Double>>){
        mData = data
        mRowSize = data.size
        mColSize = data[0].size
        shape = Pair(data.size,data[0].size)

    }
    constructor(matrixString: String){
        val listOfRowStrings: List<String> = matrixString.split("\n")
        for (row in listOfRowStrings) {
            val doubleListOfEntries: MutableList<Double> = mutableListOf()
            val listOfEntries = row.split(",")
            for (entry in listOfEntries){
                doubleListOfEntries.add(entry.toDouble())
            }
            mData.add(doubleListOfEntries)
        }
        mRowSize = mData.size
        mColSize = mData[0].size
        shape = Pair(mData.size,mData[0].size)

    }
    operator fun times(rightMatrix: Matrix): Matrix{
        if(this.mColSize != rightMatrix.mRowSize){
            throw java.lang.IllegalArgumentException("Matrix dimensions not compatible for multiplication!")
        }
        val product = Matrix(this.getRowSize(),rightMatrix.getColSize())
        for (i in 0 until mRowSize) {
            for (j in 0 until rightMatrix.mColSize) {
                for (k in 0 until mColSize) {
                    product.mData[i][j] += this.mData[i][k] * rightMatrix.mData[k][j]
                }
            }
        }
        return product
    }
    operator fun get(i: Int): MutableList<Double> {
        return mData[i]
    }
    override fun toString(): String {
        return this.mData.joinToString(separator = "\n")
    }
    fun copy(): Matrix {
        val newData: MutableList<MutableList<Double>> = mutableListOf()
        for (row in this.mData){
            val newRow: MutableList<Double> = mutableListOf()
            for (entry in row){
                newRow.add(entry)
            }
            newData.add(newRow)
        }
        return Matrix(newData)
    }
    fun getData(): MutableList<MutableList<Double>>{
        return this.mData
    }
    fun getRowSize(): Int{
        return this.mRowSize
    }
    fun getColSize(): Int{
        return this.mColSize
    }
    fun addOneToFront(){
        this.mData.add(0, MutableList(1){ 1.0 })
        mRowSize = mData.size
        mColSize = mData[0].size
        shape = Pair(mData.size,mData[0].size)
    }
    companion object{
        fun minMaxNorm(input: Matrix): Matrix {
            var min = 0.0
            var max = 0.0
            for (row in input.getData()){
                if(row[0] < min){
                    min = row[0]
                }
                if(row[0] > max){
                    max = row[0]
                }
            }

            val output = input.copy()
            for (i in 0 until input.getRowSize()) {
                output[i][0] = (input[i][0] - min) / (max - min)
            }
            return output
        }
        fun tanSigmoid(input: Matrix): Matrix {
            val output = input.copy()
            for (i in 0 until input.getRowSize()){
                output[i][0] = (2 * (1 / (1 + E.pow(-2*input[i][0]))))-1
            }
            return output
        }

        fun logSigmoid(input: Matrix): Matrix{
            val output = input.copy()
            for (i in 0 until input.getRowSize()){
                output[i][0] = (1)/(1+E.pow(-input[i][0]))
            }
            return output
        }
    }
}