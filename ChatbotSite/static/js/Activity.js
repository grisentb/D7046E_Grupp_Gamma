class Activity{
    constructor(pos, name, status){
        this.pos = pos;
        this.name = name;
        this.status = status !== "complete" ? false: true;
        this.radius = 2.5;
    }
    setStatus(status){
        this.status = status !== "complete" ? false: true;
    }
}