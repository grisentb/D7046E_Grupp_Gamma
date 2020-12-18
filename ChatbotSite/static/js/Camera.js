
class Camera{
    constructor(){
        this.pos = [0,0];
        this.theta = 0;
        this.zoom = 1.005;
        this.transFormationMatrix = this.generateTransformationMatrix();
    }

    /***
     * Update Translation Matrix, will effect the translation of each object
     * @param pos
     * @param zoom
     * @param zoom
     * @param rotation
     */
    updateTranslation(){
        this.transFormationMatrix = this.generateTransformationMatrix();
    }

    /***
     * Pass in a reference object to translate its points towards the camera.
     * @param {polygon}
     */
    translate(shape){
        let T = this.transFormationMatrix;
        let new_shape = shape.map((v)=>{
            let x = v[0];
            let y = v[1];

            let p = [x,y,1];
            //todo make this matrix multiplication clearer
            let Tp = [p[0]*T[0][0]+p[1]*T[0][1]+T[0][2],
                p[0]*T[1][0]+p[1]*T[1][1]+T[1][2]];
            return Tp;
         });
       return new_shape;

    }

    /***
     *
     * @param pos
     * @returns {*[]}
     */
    translatePos(pos)
    {
        let T = this.transFormationMatrix;
        let x = pos[0];
        let y = pos[1];

        let p = [x,y,1];
        //todo make this matrix multiplication clearer
        let Tp = [p[0]*T[0][0]+p[1]*T[0][1]+T[0][2],
            p[0]*T[1][0]+p[1]*T[1][1]+T[1][2]];
        return Tp;
    }

    /***
     * Generate Transformation/View Matrix
     * @returns {number[][]}
     */
    generateTransformationMatrix() {
        let translation = [[1,0,0],[0,1,0],[0,0,1]];
        translation[0] = [Math.cos(this.theta)*this.zoom, -Math.sin(this.theta)*this.zoom,  + this.pos[0]];
        translation[1] = [Math.sin(this.theta)*this.zoom, Math.cos(this.theta)*this.zoom, -this.zoom*this.pos[1] + this.pos[1]];
        return translation;
    }
}